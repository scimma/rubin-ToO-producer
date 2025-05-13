import astropy.table
from io import BytesIO
import math
import pytest

from forward_alerts import Skymap, LVKAlertFilter

def make_skymap(cred_area: float):
	t = astropy.table.Table()
	
	in_dens = 0.9/cred_area
	def order_area(order):
		return math.pi/(3<<(order<<1))
	def order_offset(order):
		return 4<<(2*order)
	
	# use this as the base (maximum) pixel order
	base_order = 6
	base_offset = order_offset(base_order)
	# generate the high density pixels in the region of interest
	n_in = math.ceil(cred_area/order_area(base_order))
	indices = [i+base_offset for i in range(0,n_in)]
	
	# we've rounded up the area in the region, so use that rounded up area and total probability to
	# compute the density for the outside pixels
	out_dens = (1. - n_in*order_area(base_order)*in_dens)/(4*math.pi - n_in*order_area(base_order))
	
	cur_order = base_order
	start_pixel = n_in
	n_out = 0
	while cur_order >= 0:
		# add low density pixels outside the region of interest until this order fills an integral
		# number of pixels of the next lower order
		offset = order_offset(cur_order)
		if cur_order > 0:
			pixels_to_add = 4 - start_pixel%4
			if pixels_to_add == 4:
				pixels_to_add = 0
		
		else:
			pixels_to_add = 12 - start_pixel
		print(f"order: {cur_order}, start pixel: {start_pixel}, pixels to add: {pixels_to_add}")
		indices.extend([i+offset for i in range(start_pixel, start_pixel+pixels_to_add)])
		n_out += pixels_to_add
		# set up for next iteration:
		# recompute base pixel index, and decrease order
		start_pixel += pixels_to_add
		assert start_pixel%4 == 0
		# drop two lowest bits to get parent pixel index
		start_pixel >>= 2
		cur_order -= 1
	
	# reverse entries so that low density pixels come first, to make Skymap have to sort them
	indices.reverse()
	densities = n_out*[out_dens] + n_in*[in_dens]
	assert len(densities) == len(indices)
	t["UNIQ"] = indices
	t["PROBDENSITY"] = densities
	
	buffer = BytesIO()
	t.write(buffer, format="fits")
	return buffer.getvalue()

def test_Skymap_area_for_probability():
	raw = make_skymap(4.264e-3) # ~14 sq. deg.
	map_data = astropy.table.Table.read(BytesIO(raw))
	print(f"Map has {len(map_data['UNIQ'])} pixels")
	print(map_data["PROBDENSITY"])
	print(map_data["UNIQ"])
	skymap = Skymap(map_data["PROBDENSITY"], map_data["UNIQ"])
	assert skymap.area_for_probability(0.9) >= 4.264e-3
	assert skymap.area_for_probability(0.9) <= 4.264e-3 + (math.pi/180.)**2
	
	raw = make_skymap(0.233946) # ~768 sq. deg.
	map_data = astropy.table.Table.read(BytesIO(raw))
	print(f"Map has {len(map_data['UNIQ'])} pixels")
	print(map_data["PROBDENSITY"])
	print(map_data["UNIQ"])
	skymap = Skymap(map_data["PROBDENSITY"], map_data["UNIQ"])
	assert skymap.area_for_probability(0.9) >= 0.233946
	assert skymap.area_for_probability(0.9) <= 0.233946 + (math.pi/180.)**2

def test_LVK_is_test():
	filter = LVKAlertFilter({}, None, True)
	assert filter.is_test({"superevent_id": "MS250509a"}, None)
	assert not filter.is_test({"superevent_id": "S250509a"}, None)

def test_LVK_alert_identifier():
	filter = LVKAlertFilter({}, None, True)
	test_id = "some_event_id"
	alert_time = "2025-05-14T01:04:06.789Z"
	event_time = "2025-05-14T01:02:03.456Z"
	test_alert = {
		"superevent_id": test_id,
		"alert_type": "INITIAL",
		"time_created": alert_time,
		"event": {
			"time": event_time,
		},
	}
	id_result, id_meta = filter.alert_identifier(test_alert, None)
	assert id_result == test_id
	assert id_meta["type"] == "INITIAL"
	assert id_meta["time"] == event_time

def test_LVK_overrides_previous():
	filter = LVKAlertFilter({}, None, True)
	prelim =  {"type": "PRELIMINARY"}
	init =    {"type": "INITIAL"}
	retract = {"type": "RETRACTION"}
	assert not filter.overrides_previous(init, prelim)
	assert not filter.overrides_previous(init, init)
	assert filter.overrides_previous(init, retract)

def test_LVK_should_follow_up_no_preliminary():
	filter = LVKAlertFilter({}, None, True)
	
	result, result_data = filter.should_follow_up({"alert_type": "PRELIMINARY"}, None)
	assert not result

def test_LVK_should_follow_up_bns_nsbh_merger():
	filter = LVKAlertFilter({}, None, True)
	
	case_B_merger = {
		"alert_type": "INITIAL",
		"event": {
			"skymap": make_skymap(0.022846), # 75 sq. deg.
			"far": 1e-8,
			"classification": {
				"BNS": 0.75,
				"NSBH": 0.2,
				"BBH": 0.04,
				"Noise": 0.01,
			},
			"properties": {
				"HasNS": 0.95,
				"HasRemnant": 0.75,
				"HasMassGap": 0.08,
				"HasSSM": 0.3,
			},
		},
	}
	result, result_data = filter.should_follow_up(case_B_merger, None)
	assert result
	assert result_data["type"] == "GW_case_B"
	
	case_D_merger = {
		"alert_type": "INITIAL",
		"event": {
			"skymap": make_skymap(0.121846), # 400 sq. deg.
			"far": 1e-8,
			"classification": {
				"BNS": 0.75,
				"NSBH": 0.2,
				"BBH": 0.04,
				"Noise": 0.01,
			},
			"properties": {
				"HasNS": 0.95,
				"HasRemnant": 0.75,
				"HasMassGap": 0.08,
				"HasSSM": 0.3,
			},
		},
	}
	result, result_data = filter.should_follow_up(case_D_merger, None)
	assert result
	assert result_data["type"] == "GW_case_D"
	
	badly_localized_merger = {
		"alert_type": "INITIAL",
		"event": {
			"skymap": make_skymap(0.243692), # 800 sq. deg.
			"far": 1e-8,
			"classification": {
				"BNS": 0.75,
				"NSBH": 0.2,
				"BBH": 0.04,
				"Noise": 0.01,
			},
			"properties": {
				"HasNS": 0.95,
				"HasRemnant": 0.75,
				"HasMassGap": 0.08,
				"HasSSM": 0.3,
			},
		},
	}
	result, result_data = filter.should_follow_up(badly_localized_merger, None)
	assert not result
	
	low_ns_prob = {
		"alert_type": "INITIAL",
		"event": {
			"skymap": make_skymap(0.121846), # 400 sq. deg.
			"far": 1e-8,
			"classification": {
				"BNS": 0.05,
				"NSBH": 0.2,
				"BBH": 0.74,
				"Noise": 0.01,
			},
			"properties": {
				"HasNS": 0.25,
				"HasRemnant": 0.05,
				"HasMassGap": 0.08,
				"HasSSM": 0.3,
			},
		},
	}
	result, result_data = filter.should_follow_up(low_ns_prob, None)
	assert not result
	
	low_remnant_prob = {
		"alert_type": "INITIAL",
		"event": {
			"skymap": make_skymap(0.022846), # 75 sq. deg.
			"far": 1e-8,
			"classification": {
				"BNS": 0.05,
				"NSBH": 0.9,
				"BBH": 0.04,
				"Noise": 0.01,
			},
			"properties": {
				"HasNS": 0.95,
				"HasRemnant": 0.05,
				"HasMassGap": 0.08,
				"HasSSM": 0.3,
			},
		},
	}
	result, result_data = filter.should_follow_up(low_remnant_prob, None)
	assert not result
	
	high_far = {
		"alert_type": "INITIAL",
		"event": {
			"skymap": make_skymap(0.022846), # 75 sq. deg.
			"far": 4e-8,
			"classification": {
				"BNS": 0.75,
				"NSBH": 0.2,
				"BBH": 0.04,
				"Noise": 0.01,
			},
			"properties": {
				"HasNS": 0.75,
				"HasRemnant": 0.75,
				"HasMassGap": 0.08,
				"HasSSM": 0.3,
			},
		},
	}
	result, result_data = filter.should_follow_up(high_far, None)
	assert not result

def test_LVK_should_follow_up_lensed_bns_merger():
	filter = LVKAlertFilter({}, None, True)
	
	case_A_merger = {
		"alert_type": "INITIAL",
		"event": {
			"skymap": make_skymap(0.243692), # 800 sq. deg.
			"far": 1e-8,
			"classification": {
				"BNS": 0.85,
				"NSBH": 0.05,
				"BBH": 0.04,
				"Noise": 0.01,
			},
			"properties": {
				"HasNS": 0.9,
				"HasRemnant": 0.5,
				"HasMassGap": 0.95,
				"HasSSM": 0.1,
			},
		},
	}
	result, result_data = filter.should_follow_up(case_A_merger, None)
	assert result
	assert result_data["type"] == "lensed_BNS_case_A"
	
	case_B_merger = {
		"alert_type": "INITIAL",
		"event": {
			"skymap": make_skymap(3.655409e-3), # 12 sq. deg.
			"far": 1e-8,
			"classification": {
				"BNS": 0.85,
				"NSBH": 0.05,
				"BBH": 0.04,
				"Noise": 0.01,
			},
			"properties": {
				"HasNS": 0.9,
				"HasRemnant": 0.5,
				"HasMassGap": 0.95,
				"HasSSM": 0.1,
			},
		},
	}
	result, result_data = filter.should_follow_up(case_B_merger, None)
	assert result
	assert result_data["type"] == "lensed_BNS_case_B"
	
	low_mass_gap_prob = {
		"alert_type": "INITIAL",
		"event": {
			"skymap": make_skymap(0.243692), # 800 sq. deg.
			"far": 1e-8,
			"classification": {
				"BNS": 0.85,
				"NSBH": 0.05,
				"BBH": 0.04,
				"Noise": 0.01,
			},
			"properties": {
				"HasNS": 0.9,
				"HasRemnant": 0.5,
				"HasMassGap": 0.85,
				"HasSSM": 0.1,
			},
		},
	}
	result, result_data = filter.should_follow_up(low_mass_gap_prob, None)
	assert not result
	
	high_nsbh_prob = {
		"alert_type": "INITIAL",
		"event": {
			"skymap": make_skymap(0.243692), # 800 sq. deg.
			"far": 1e-8,
			"classification": {
				"BNS": 0.75,
				"NSBH": 0.15,
				"BBH": 0.04,
				"Noise": 0.01,
			},
			"properties": {
				"HasNS": 0.9,
				"HasRemnant": 0.5,
				"HasMassGap": 0.95,
				"HasSSM": 0.1,
			},
		},
	}
	result, result_data = filter.should_follow_up(high_nsbh_prob, None)
	assert not result
	
	high_far = {
		"alert_type": "INITIAL",
		"event": {
			"skymap": make_skymap(0.243692), # 800 sq. deg.
			"far": 4e-8,
			"classification": {
				"BNS": 0.85,
				"NSBH": 0.05,
				"BBH": 0.04,
				"Noise": 0.01,
			},
			"properties": {
				"HasNS": 0.9,
				"HasRemnant": 0.5,
				"HasMassGap": 0.95,
				"HasSSM": 0.1,
			},
		},
	}
	result, result_data = filter.should_follow_up(high_far, None)
	assert not result
	
	poor_localization = {
		"alert_type": "INITIAL",
		"event": {
			"skymap": make_skymap(0.304617), # 1000 sq. deg.
			"far": 1e-8,
			"classification": {
				"BNS": 0.85,
				"NSBH": 0.05,
				"BBH": 0.04,
				"Noise": 0.01,
			},
			"properties": {
				"HasNS": 0.9,
				"HasRemnant": 0.5,
				"HasMassGap": 0.85,
				"HasSSM": 0.1,
			},
		},
	}
	result, result_data = filter.should_follow_up(poor_localization, None)
	assert not result
