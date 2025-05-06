#!/usr/bin/env python3

import argparse
from astropy.io import fits
import astropy.table
import copy
import fastavro
import hop
from io import BytesIO
import json
import logging
import math
import numpy
import orjson
import requests
import sys
import yaml
import zlib

logger = logging.getLogger("ToO Alert Producer")

def load_yaml_config(file_path, config):
	"""Load settings from file_path and merge into config"""
	logger.debug(f"Loading configuration from {file_path}")
	try:
		with open(file_path) as f:
			config_data = yaml.safe_load(f)
		for key, value in config_data.items():
			if '-' in key:
				key = key.replace('-','_')
			setattr(config, key, value)
	except Exception as e:
		raise RuntimeError(f"Unable to read YAML config from {file_path}") from e

class LoadYamlConfig(argparse.Action):
	def __init__(self, **kwargs):
		if "default" in kwargs:
			del kwargs["default"]
		kwargs["required"] = False
		super().__init__(**kwargs)
	
	def __call__(self, parser, namespace, file_path, option_string=None):
		delattr(namespace, self.dest)
		load_yaml_config(file_path, namespace)

class KahanAdder:
	def __init__(self):
		self.sum = 0.0
		self.comp = 0.0
		
	def __float__(self):
		return self.sum
	
	def __iadd__(self, value):
		y = value - self.comp
		t = self.sum + y
		self.comp = (t - self.sum) - y
		self.sum = t
		return self
	
	def __eq__(self, value):
		return self.sum == value
	
	def __ne__(self, value):
		return self.sum != value
	
	def __lt__(self, value):
		return self.sum < value
	
	def __le__(self, value):
		return self.sum <= value
	
	def __gt__(self, value):
		return self.sum > value
	
	def __ge__(self, value):
		return self.sum >= value

class Skymap:
	def __init__(self, densities, u_indices, drop_trivial_probabilities: bool=True):
		"""
		Construct a map from arrays of probability densities and healpix UNIQ ordering indices.
		Data in the map will be re-ordered with entries sorted in order of decreasing density.
		
		Args:
		    drop_trivial_probabilities: If set, all of the pixels whose contributions to the total
		                                probability are so small that the sum of probabilities from
		                                higher probability pixels is one to within floating-point
		                                epsilon have their probability densities set to zero, making
		                                the map more compressible.
		"""
		self.data = numpy.array(sorted(zip(densities, u_indices), key=lambda entry: -entry[0]), 
		                        dtype=[("prob_density", "f8"), ("uniq_index", "i8")])
		
		# Many entries contribute tiny probabilities which should be unimportant.
		# We seek to replace these with zeros, which will hopefully form large, compressible
		# runs when we are forced to make a single (maximal) order map.
		# Sum probability until the total reaches one. This will generally occur before the end of
		# the data due to limited floating-point precision, although we use Kahan summation to avoid
		# it happening unduly quickly; values after this can be considered irrelevant since they
		# are each too small to cause the sum to increase further (although it might still
		# increase it if the Kahan algorithm were continued and they collectively exceed
		# floating-point epsilon).
		summed_prob=KahanAdder()
		comp=0
		# A cache of the areas of pixels at all relevant orders
		self.pixel_areas={}
		for entry in self.data:
			order=math.floor(math.log2(entry[1]/4)/2)
			if drop_trivial_probabilities and summed_prob>=1.0:
				entry[0]=0
				continue;
			if order not in self.pixel_areas:
				self.pixel_areas[order] = math.pi/(3<<(order<<1))
			area = self.pixel_areas[order]
			prob=entry[0] * area
			summed_prob+=prob
		#print("Order range:",min(self.pixel_areas.keys()),'-',max(self.pixel_areas.keys()))
	
	def area_for_probability(self, target_probability: float):
		"""
		Compute the area on the sky subtended by the highest density portion of the map which
		sums to the target probability.
		"""
		summed_prob=KahanAdder()
		summed_area=KahanAdder()
		for p_dens, u_idx in self.data:
			order=math.floor(math.log2(u_idx/4)/2)
			area = self.pixel_areas[order]
			prob=p_dens * area
			summed_prob+=prob
			if summed_prob < target_probability:
				summed_area+=area
			else:
				break
		return summed_area.sum
	
	def make_flat_map(self):
		# first, figure out how many pixels it must be from its order, which must be the maximum
		# order in the original map
		flat_order=max(self.pixel_areas.keys())
		flat_n_npixels=12<<(flat_order<<1)
		
		flat_map = numpy.zeros(flat_n_npixels)
		
		# next, iterate over the original map, and copy its data into all corresponding pixels
		# of the flat map
		for p_dens, u_idx in self.data:
			# u_idx is the original pixel's UNIQ index
			order = math.floor(math.log2(u_idx/4)/2)
			# the original pixel's NESTED ordering index, within its order
			n_idx = u_idx - (4<<(2*order))
			# each unit of increase in order adds two bits to the child pixel indices
			min_idx = n_idx << (2 * (flat_order - order))
			# the increase in order splits the original pixel into this many pixels
			idx_range = 1 << (2 * (flat_order - order))
			flat_map[min_idx:(min_idx+idx_range)] = p_dens
		
		return flat_map
	
	def make_flat_binary_map(self, target_probability: float, target_order = None):
		# first, figure out how many pixels it must be from its order, which must be the maximum
		# order in the original map
		if target_order is None:
			target_order=max(self.pixel_areas.keys())
		flat_n_npixels=12<<(target_order<<1)
		print(f"Making order {target_order} map with {flat_n_npixels} pixels")
		
		flat_map = numpy.zeros(flat_n_npixels, dtype=numpy.int8)
		
		# next, iterate over the original map, and writing ones for all pixels which are above the
		# target threshold
		summed_prob=KahanAdder()
		for p_dens, u_idx in self.data:
			# u_idx is the original pixel's UNIQ index
			order = math.floor(math.log2(u_idx/4)/2)
			
			# the original pixel's NESTED ordering index, within its order
			n_idx = u_idx - (4<<(2*order))
			if order < target_order:
				# each unit of increase in order adds two bits to the child pixel indices
				min_idx = n_idx << (2 * (target_order - order))
				# the increase in order splits the original pixel into this many pixels
				idx_range = 1 << (2 * (target_order - order))
				flat_map[min_idx:(min_idx+idx_range)] = 1.0
			else:
				# throw away two bits for each unit of difference in order to find parent pixel
				p_idx = n_idx >> (2 * (order - target_order))
				flat_map[p_idx] = 1.0
			
			area = self.pixel_areas[order]
			prob=p_dens * area
			summed_prob+=prob
			
			if summed_prob >= target_probability:
				break
		
		return flat_map

def write_json(records, compressed: bool=False):
	buf=orjson.dumps(records, option=orjson.OPT_SERIALIZE_NUMPY)
	if(compressed):
		zbuf=zlib.compress(buf.encode("utf-8"), level=9)
		return zbuf
	else:
		return buf


# not actually a class, just a wrapper function for working the annoying hop.io.Stream middleman
def KafkaConsumer(url: str, *args, **kwargs):
	return hop.io.Stream().open(url, mode='r', *args, **kwargs)


class FileConsumer:
	def __init__(self, paths):
		self.paths = paths
	
	def read(self, metadata: bool=False, autocommit: bool=False):
		"""
		Args:
		    metadata: Whether to return metadata with each message. Produces minimal, kafka-like
		              information, in particular all messages will be labeled as coming from a topic
		              named "all".
		    autocommit: Ignored, present only for consistency with the Kafka consumer.
		"""
		counter = 0
		for filepath in self.paths:
			logger.info("Reading input from %s", filepath)
			# currently we assume that all input files are Avro
			msg = hop.models.AvroBlob.load_file(filepath)
			if metadata:
				m = hop.io.Metadata(topic="all", partition=0, offset=counter, timestamp=0, key=b"", 
				                    headers=[], _raw=None)
				yield msg, m
			else:
				yield msg
			counter += 1
	
	def mark_done(self, *args):
		"""
		Does nothing, but keeps compatibility with the kafka consumer interface
		"""
		pass


class AlertSender:
	def __init__(self, output_schema):
		self.schema = output_schema
	
	def send(self, data: dict, test: bool=False):
		raise NotImplementedError


class StdoutSender(AlertSender):
	def __init__(self, output_schema):
		super().__init__(output_schema)
	
	def send(self, data: dict, test: bool=False):
		if test:
			print("*** TEST ALERT ***")
		print(data)
		sys.stdout.flush()


class KafkaSender(AlertSender):
	def __init__(self, output_schema, url):
		super().__init__(output_schema)
		self.producer = hop.io.Stream().open(url, 'w')
	
	def send(self, data: dict, test: bool=False):
		msg = hop.models.AvroBlob(content=data, schema=self.schema)
		self.producer.write(msg, test=test)
		self.producer.flush()  # mesage rate should be low, nudge librdkafka not to wait for more


class ConfluentRESTSender(AlertSender):
	def __init__(self, output_schema, url):
		super().__init__(output_schema)
		self.url = url
	
	def send(self, data: dict, test: bool=False):
		raw_body = {"value_schema": self.schema,
		            "records": [{"value": data}],
		            }
		request_body = write_json(raw_body, compressed=True)
		
		additional_headers = {"Content-Type": "application/vnd.kafka.json.v2+json",
		                      "Content-Encoding": "gzip",
		                      }
		resp = requests.post(url, data=request_body, headers=additional_headers)
		if resp.status_code != 200:
			logging.error(f"POST to Confluent REST Proxy failed ({resp.status_code}): "
			              f"{resp.content}")
			# TODO: figure out what if anything else we should do about errors


class AlertFilter:
	def __init__(self, history: dict, sender: AlertSender, allow_tests: bool):
		"""
		Args:
			history: A mapping of alert identifiers to alert details, used for filtering duplicates.
			         Alerts may be duplictaed both due to data transport issues, or due to multiple
			         alerts being sent for the same event, which may or may not be of superseding
			         interest to this system.
			allow_tests: If true, alerts marked as tests are fully processed, otherwise they are
			             dropped.
		"""
		self.history = history
		self.sender = sender
		self.allow_tests = allow_tests
	
	def is_test(self, message, metadata):
		"""
		Determine whether a given alert message is a test message, which may be ignored depending on
		operating mode.
		The default implementation simply checks the HOPSKOTCH _test header; subclasses should add
		any additional checks which are relevant for their message format(s).
		
		Args:
			message: The decoded message object
			metadata: Transport metadata for the message
		"""
		if metadata is None:
			return False
		for header in metadata.headers:
			if header[0] == "_test":
				return True
		return False
	
	def alert_identifier(self, message, metadata):
		"""
		Extract a unique identifier for an alert.
		The default implementation simply uses the HOPSKOTCH message UUID, however it is better to
		use identifiers with more sematics specific to an alert format/type.
		
		Args:
			message: The decoded message object
			metadata: Transport metadata for the message
		Return: A 2-tuple of the event identifier and any associated metadata which may be needed to
		        determine whether another alert with the same identifier supersede the one(s)
		        previously seen. The latter may be or include something a message maturity/lifecycle
		         type code, e.g. "preliminary", "normal", "retraction", etc., an alert version
		         number, or an alert time.
		"""
		for header in metadata.headers:
			if header[0] == "_id":
				return header[1], None
		return None, None
	
	def overrides_previous(self, old_meta, new_meta):
		"""
		Two messages with the same identifier may or may not be exact duplicates, and if not the new
		message may or may not supersede the previous version. Forexample, a retraction may mean
		that any scheduling for the previous version(s) should be abandoned.
		
		The default implementation always indicates that subsequent alerts are ignored.
		"""
		return False
	
	def should_follow_up(self, message, metadata):
		"""
		The core filtering routine: Decides whether the alert should be followed up.
		
		the default implementation rejects all alerts.
		
		Args:
			message: The decoded message object
			metadata: Transport metadata for the message
		Return: A 2-tuple of a boolean value indicating whether follow up is indicated and any
		        useful data produced during the check which should be used for building the message
		        to send to the scheduler.
		"""
		return False, None
	
	def generate_scheduling_data(self, message, metadata, alert_data):
		"""
		Generate whatever data should be sent to the scheduler for this alert (which is assumed to
		have passed filtering). 
		Args:
			message: The decoded message object
			metadata: Transport metadata for the message
			alert_data: Any data derived from the message by should_follow_up
		Return: A dictionary of data to be sent.
		"""
		raise NotImplementedError
	
	def process(self, message, metadata):
		is_test = self.is_test(message, metadata)
		if not self.allow_tests and is_test:
			logger.info("Alert is a test: ignoring")
			return False
		
		alert_id, id_meta = self.alert_identifier(message, metadata)
		logger.info(f"Alert ID is {alert_id}; metadata: {id_meta}")
		
		passes, alert_data = self.should_follow_up(message, metadata)
		
		if not passes:
			return False
		
		# handle duplicates
		is_update = False
		if alert_id is not None:
			if alert_id in self.history:
				logger.info(f"This alert has been seen before")
				if not self.overrides_previous(self.history[alert_id], id_meta):
					logger.info(f"This alert message does not override the previous")
					return False
				else:
					logger.info(f"This alert message overrides the previous")
					is_update = True
			self.history[alert_id] = id_meta
		
		scheduling_data = self.generate_scheduling_data(message, metadata, alert_data)
		scheduling_data["is_test"] = is_test
		scheduling_data["is_update"] = is_update
		
		self.sender.send(scheduling_data, test=is_test)
		return True


class LVKAlertFilter(AlertFilter):
	def __init__(self, history: dict, sender: AlertSender, allow_tests: bool):
		super().__init__(history, sender, allow_tests)
	
	def is_test(self, message, metadata):
		if super().is_test(message, metadata):
			return True
		# "Prefix: S for normal candidates and MS or TS for mock or test events, respectively"
		return message["superevent_id"][0] != "S"
	
	def alert_identifier(self, message, metadata):
		return message["superevent_id"], \
		       {"type": message["alert_type"], "time": message["time_created"]}
	
	def overrides_previous(self, old_meta, new_meta):
		# Retractions override previous alerts.
		# If there were an interest in handling updates, etc., logic should be added here.
		return old_meta["type"] == "INITIAL" and new_meta["type"] == "RETRACTION"
	
	def should_follow_up(self, message, metadata):
		if message["alert_type"] != "INITIAL":
			return False, {}
	
		raw_map=astropy.table.Table.read(BytesIO(message["event"]["skymap"]))
		skymap = Skymap(raw_map["PROBDENSITY"], raw_map["UNIQ"])
		mean_dist = raw_map.meta.get("DISTMEAN", -1.0)
		prob_area = skymap.area_for_probability(0.9)
		
		logger.info(f"LVK alert with 90% probability area of {prob_area} sr")
		logger.info(f"    Mean distance: {mean_dist} Mpc")
		
		result_data = {"skymap": skymap, "90%_area": prob_area}
		passes = False
		
		# Binary Neutron Star Mergers and Neutron Star - Black Hole Mergers
		# Requirements:
		# - "Only trigger on an Initial map, do not trigger on Preliminary"
		# - "The probability of being BNS or NS-BH should be greater than 90%: BNS+NS-BH>=0.9"
		# - "False alarm rate less than 1 per 1 year: FAR < 1.6e-08 Hz"
		# - "90% sky area less than 500 square degrees" (500 deg^2 = 0.152308 sr)
		# - "For NS-BH events, require that there is a good probability that mass has been 
		#    "ejected (these numbers will be changed based on O4 results and O5 projections): 
		#    HasNS >= 0.5 and HasRemnant >= 0.5"
		#
		# Further categorization:
		# - Gold: 90% area < 100 square degrees (0.030461 sr)
		# - Silver: 90% area < 500 square degrees (0.152308 sr)
		if message["alert_type"] == "INITIAL" and \
		  (message["event"]["classification"]["BNS"] + 
		   message["event"]["classification"]["NSBH"]) >= 0.9 and \
		  message["event"]["far"] < 1.6e-08 and \
		  prob_area < 0.152308 and \
		  message["event"]["properties"]["HasNS"] >= 0.5 and \
		  message["event"]["properties"]["HasRemnant"] >= 0.5:
			passes = True
			quality = "gold" if prob_area < 0.030461 else "silver"
			result_data["type"] = f"BNS/NSBH-{quality}"
			logger.info(f"LVK alert meets criteria for {quality} BNS or NSBH merger")
		
		# TODO: 'Very large skymaps'
		# Requirements:
		# - Only trigger on Initial maps?
		# - "The probability of being BNS or NS-BH should be greater than 90%: BNS+NS-BH>=0.9"
		# - False alram rate requirement?
		# - 90% area > 1000 square degrees (0.304617 sr)
		# - Remnant requirement?
		if message["alert_type"] == "INITIAL" and \
		  (message["event"]["classification"]["BNS"] + 
		   message["event"]["classification"]["NSBH"]) >= 0.9 and \
		   prob_area >= 0.304617:
			logger.warning("Alert might pass Very Large Skymap conditions, "
			               "but these are not definitely implemented")
		
		# Gravitationally lensed Binary Neutron Star mergers
		# Requirements:
		# - "Only trigger on an Initial human-vetted GW detections"
		# - "probability that the GW source includes one or more compact objects in the range
		#   3 – 5 M☉ of no less than 90%: p(HasMassGap)>=0.9"
		# - probability that the GW source is a NS-BH merger of less than 10% : p(NS-BH)<0.1
		# - "False alarm rate less than 1 per 1 year: FAR < 1.6e-08 Hz"
		# - "90% credible GW sky localization of no more than 900 degree^2"
		#   (900 deg^2 = 0.2741556)
		#
		# Further categorization:
		# - Gold: 90% area < 15 square degrees (4.569261e-3 sr)
		# - Silver: 90% area < 900 square degrees (0.2741556 sr)
		if message["alert_type"] == "INITIAL" and \
		  message["event"]["properties"]["HasMassGap"] >= 0.9 and \
		  message["event"]["classification"]["NSBH"] < 0.1 and \
		  message["event"]["far"] < 1.6e-08 and \
		  prob_area < 0.2741556:
			passes = True
			quality = "gold" if prob_area < 4.569261e-3 else "silver"
			result_data["type"] = f"LensedBNS-{quality}"
			logger.info(f"LVK alert meets criteria for {quality} lensed BNS merger")

		# TODO: implement
		# Black Hole-Black Hole Mergers
		# Requirements:
		# - "90% sky area less than 20 square degrees"
		# - "distance <6 Gpc"
		# - "total mass>50 M☉"
		#	"Unless in O5 the LVK will release a flag for massive binaries, we will select 
		#	events based on estimated mass as in
		#	https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.251102"
		
		# TODO: implement unidentified source alerts
		# Further categorization:
		# - Gold: 90% area < 100 square degrees (0.030461 sr)
		# - Silver: 90% area < 500 square degrees (0.152308 sr)
		if not passes and \
		   message["alert_type"] == "INITIAL" and \
		   prob_area < 0.152308:
			quality = "gold" if prob_area < 4.569261e-3 else "silver"
			logger.warning(f"Alert might pass Unidentified Source conditions with {quality} "
			               "quality, but these are not definitely implemented")
		
		return passes, result_data
	
	def generate_scheduling_data(self, message, metadata, alert_data):
		target_order = 5
		flat_map = alert_data["skymap"].make_flat_binary_map(0.9, target_order)
		return {"alert_type": alert_data["type"],
		        "event_trigger_timestamp": message["event"]["time"],
		        "reward_map": flat_map,
		        "reward_map_nside": 1<<target_order,
		        }


input_constructors = {
	"files": FileConsumer,
	"kafka": KafkaConsumer,
}

filter_constructors = {
	"lvk_gw": LVKAlertFilter
}

output_constructors = {
	"stdout": StdoutSender,
	"kafka": KafkaSender,
	"confluent_rest": ConfluentRESTSender,
}

logging.basicConfig(level=logging.INFO) # TODO: make configurable

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--config-file", help="read configuration from a YAML file",
                    action=LoadYamlConfig)
parser.add_argument("--allow-tests", type=bool, default=False,
                    help="whether to process or discard test alerts")
parser.add_argument("--input-type", type=str, choices=input_constructors.keys(), default="files",
                    help="the mechanism to use for reading input alerts")
parser.add_argument("--input-options", type=json.loads, default={}, 
                    help="settings for the input consumer")
parser.add_argument("--filters", type=json.loads, default={}, 
                    help="mapping of topic names to filter types")
parser.add_argument("--output-type", type=str, choices=output_constructors.keys(), default="stdout",
                    help="the mechanism to use for sending passing alert data")
parser.add_argument("--output-options", type=json.loads, default={}, 
                    help="settings for the output sender")
parser.add_argument("input_files", nargs='*', help="files to be read with the file consumer")

config = parser.parse_args()

if config.input_type not in input_constructors:
	logger.fatal(f"Unrecognized input type: {config.input_type}")
	exit(1)
if config.output_type not in output_constructors:
	logger.fatal(f"Unrecognized output type: {config.output_type}")
	exit(1)

with open("output_schema.json") as schema_file:
	output_schema = json.load(schema_file)

# TODO: The history grows unboundedly. How/when should items be removed?
history = {}

# Hack: for a few special cases, move config data around
if config.input_type == "files":
	if "paths" in config.input_options:
		config.input_options["paths"].extend(config.input_files)
	else:
		config.input_options["paths"] = config.input_files
if config.input_type == "kafka":
	config.input_options["ignoretest"] = not config.allow_tests

consumer = input_constructors[config.input_type](**config.input_options)
sender = output_constructors[config.output_type](output_schema=output_schema, 
                                                 **config.output_options)
filters = {}
for topic, filter in config.filters.items():
	filters[topic] = filter_constructors[filter](history, sender, allow_tests=config.allow_tests)

for message, metadata in consumer.read(metadata=True, autocommit=False):
	# TODO: this structure assumes that all messages are avro, and should be generalized
	for record in (message.content if not message.single_record else [message.content]):
		if metadata.topic not in filters:
			logger.error(f"Message metadata claims it is from unexpected topic '{metadata.topic}'")
			continue
		try:
			filters[metadata.topic].process(record, metadata)
			consumer.mark_done(metadata)
		except Exception as e:
			logger.error(f"Error processing alert: {e}\nDropping and continuing with next")
