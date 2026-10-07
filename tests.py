import astropy.table
from astropy.coordinates import Galactic, ICRS, get_sun
from astropy.time import Time
import astropy.units as u
from astropy.utils import iers
from astropy_healpix import HEALPix
import gzip
from io import BytesIO
import math
import numpy
import pytest

import forward_alerts
from forward_alerts import Skymap, LVKAlertFilter, IceCubeAlertFilter, RecentEventCache, \
                           trim_to_observable, observable_mask, DEG2_TO_SR, FAR_1_PER_YR, \
                           FAR_1_PER_3YR, PROB_MAP_ORDER, PROB_MAP_PIXEL_AREA_DEG2

# Do not try to download Earth orientation data while testing
iers.conf.auto_download = False

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

# A fixed event time for tests which depend on observability
T0 = "2026-11-15T03:00:00.000Z"
# A position which is observable from Rubin within a day of T0, far from the Galactic plane
HIGH_B_POS = (60., -30.)
# A position on the Galactic plane (l=270, b=0), which is observable within a day of T0
_plane = Galactic(l=270*u.deg, b=0*u.deg).transform_to(ICRS())
PLANE_POS = (_plane.ra.deg, _plane.dec.deg)
# The position of the Sun at T0, which is not observable at night
_sun = get_sun(Time(T0)).transform_to(ICRS())
SUN_POS = (_sun.ra.deg, _sun.dec.deg)

def _nearest_pixels(order: int, n: int, ra: float, dec: float):
	"""The NESTED indices of the n pixels at order whose centres are nearest (ra, dec)"""
	hp = HEALPix(nside=1<<order, order="nested", frame=ICRS())
	centres = hp.healpix_to_skycoord(numpy.arange(hp.npix))
	centre = ICRS(ra=ra*u.deg, dec=dec*u.deg)
	separations = centres.separation(centre).rad
	return numpy.argsort(separations, kind="stable")[:n], hp.npix

def make_skymap_at(cred_area_deg2: float, ra: float, dec: float, cred: float=0.9001):
	"""
	Make an order 6 multi-order (UNIQ) map whose 90% credible region is the roughly circular region
	of the given area around (ra, dec), with the remaining probability spread over the rest of the sky.
	The region holds slightly more than 90% of the probability, so that rounding cannot extend the
	90% credible region outside it.
	"""
	order = 6
	pixel_area = math.pi/(3<<(order<<1))
	n_in = math.ceil(cred_area_deg2*DEG2_TO_SR/pixel_area)
	in_pixels, npix = _nearest_pixels(order, n_in, ra, dec)
	in_dens = cred/(n_in*pixel_area)
	out_dens = (1. - cred)/((npix - n_in)*pixel_area)
	densities = numpy.full(npix, out_dens)
	densities[in_pixels] = in_dens
	t = astropy.table.Table()
	t["UNIQ"] = (4<<(2*order)) + numpy.arange(npix)
	t["PROBDENSITY"] = densities
	buffer = BytesIO()
	t.write(buffer, format="fits")
	return buffer.getvalue()

def skymap_from_bytes(raw):
	map_data = astropy.table.Table.read(BytesIO(raw))
	return Skymap(map_data["PROBDENSITY"], map_data["UNIQ"])

# Chirp mass bin edges used by the LVK, as seen in real mchirp_source.json files
MCHIRP_EDGES = [0.1, 0.87, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9, 2.1, 2.3, 3.0, 5.5, 11.0, 22.0,
                44.0, 88.0, 1000.0]

def mass_data(bin_probs: dict):
	"""Make chirp mass data from a mapping of lower bin edges to probabilities"""
	probs = [bin_probs.get(edge, 0.0) for edge in MCHIRP_EDGES[:-1]]
	return MCHIRP_EDGES, probs

def lvk_filter(monkeypatch=None, masses=None, cache=None):
	filter = LVKAlertFilter({}, None, True, event_cache=cache if cache is not None else RecentEventCache())
	if masses is not None:
		monkeypatch.setattr(filter, "get_chirp_mass_estimate", lambda message, metadata: masses)
	return filter

def lvk_alert(area_deg2: float, pos=HIGH_B_POS, far: float=1e-8, classification=None,
              properties=None, search: str="AllSky", alert_type: str="INITIAL"):
	if classification is None:
		classification = {"BNS": 0.75, "NSBH": 0.2, "BBH": 0.04, "Noise": 0.01}
	if properties is None:
		properties = {"HasNS": 0.95, "HasRemnant": 0.75, "HasMassGap": 0.08, "HasSSM": 0.3}
	return {
		"alert_type": alert_type,
		"superevent_id": "S261115a",
		"event": {
			"time": T0,
			"skymap": make_skymap_at(area_deg2, *pos),
			"far": far,
			"search": search,
			"instruments": ["H1", "L1"],
			"classification": classification,
			"properties": properties,
		},
	}

NO_NS_CLASSIFICATION = {"BNS": 0.0, "NSBH": 0.0, "BBH": 0.99, "Noise": 0.01}
NO_NS_PROPERTIES = {"HasNS": 0.0, "HasRemnant": 0.0, "HasMassGap": 0.0, "HasSSM": 0.0}

def test_Skymap_area_for_probability():
	raw = make_skymap(4.264e-3) # ~14 sq. deg.
	map_data = astropy.table.Table.read(BytesIO(raw))
	print(f"Map has {len(map_data['UNIQ'])} pixels")
	print(map_data["PROBDENSITY"])
	print(map_data["UNIQ"])
	skymap = Skymap(map_data["PROBDENSITY"], map_data["UNIQ"])
	assert skymap.area_for_probability(0.9)[0] >= 4.264e-3
	assert skymap.area_for_probability(0.9)[0] <= 4.264e-3 + (math.pi/180.)**2
	
	raw = make_skymap(0.233946) # ~768 sq. deg.
	map_data = astropy.table.Table.read(BytesIO(raw))
	print(f"Map has {len(map_data['UNIQ'])} pixels")
	print(map_data["PROBDENSITY"])
	print(map_data["UNIQ"])
	skymap = Skymap(map_data["PROBDENSITY"], map_data["UNIQ"])
	assert skymap.area_for_probability(0.9)[0] >= 0.233946
	assert skymap.area_for_probability(0.9)[0] <= 0.233946 + (math.pi/180.)**2

def test_Skymap_flat_prob_map():
	# multi-order map, with pixels both finer and coarser than the target orders
	skymap = skymap_from_bytes(make_skymap(0.121846))
	for order in (3, PROB_MAP_ORDER):
		prob = skymap.flat_prob_map(order)
		assert len(prob) == 12<<(2*order)
		assert prob.sum() == pytest.approx(1.0, abs=1e-9)
	# the high-density region is the start of the NESTED ordering
	prob = skymap.flat_prob_map(PROB_MAP_ORDER)
	assert prob[0] > prob[-1]

def test_observable_mask():
	mask = observable_mask(1<<PROB_MAP_ORDER, T0)
	assert len(mask) == 12<<(2*PROB_MAP_ORDER)
	assert forward_alerts._observable(*HIGH_B_POS, T0)[0]
	assert not forward_alerts._observable(*SUN_POS, T0)[0]
	# never rises above 30 degrees from Rubin
	assert not forward_alerts._observable(0., 70., T0)[0]
	# a zero-length window during the day contains no night time
	assert not forward_alerts._observable(*HIGH_B_POS, "2026-11-15T18:00:00Z", window_h=0)[0]

def test_trim_to_observable():
	skymap = skymap_from_bytes(make_skymap_at(100., *HIGH_B_POS))
	mask, area, prob, high_b_prob = trim_to_observable(skymap, T0, 1500.)
	assert area == pytest.approx(100., abs=2.)
	assert prob == pytest.approx(0.9, abs=0.01)
	assert high_b_prob == pytest.approx(prob)
	
	# a large region is trimmed to the cap, and probabilities are not renormalised
	skymap = skymap_from_bytes(make_skymap_at(2500., *HIGH_B_POS))
	mask, area, prob, high_b_prob = trim_to_observable(skymap, T0, 1500.)
	assert area <= 1500.
	assert area > 1500. - PROB_MAP_PIXEL_AREA_DEG2
	assert numpy.count_nonzero(mask)*PROB_MAP_PIXEL_AREA_DEG2 == pytest.approx(area)
	assert prob < 0.9*1500./2500. + 0.01
	
	# Galactic plane region has (almost) no probability at high latitude
	skymap = skymap_from_bytes(make_skymap_at(100., *PLANE_POS))
	mask, area, prob, high_b_prob = trim_to_observable(skymap, T0, 1500.)
	assert prob > 0.5
	assert high_b_prob < 0.01
	
	# region around the Sun is not observable
	skymap = skymap_from_bytes(make_skymap_at(100., *SUN_POS))
	mask, area, prob, high_b_prob = trim_to_observable(skymap, T0, 1500.)
	assert not mask.any()
	assert area == 0

def test_LVK_prob_in_range():
	filter = lvk_filter()
	masses = mass_data({2.3: 0.5, 3.0: 0.3, 5.5: 0.1, 22.0: 0.05, 44.0: 0.05})
	assert filter.prob_in_range(None, 0., 2.3) == 0.
	assert filter.prob_in_range(masses, 2.3, 5.5) == pytest.approx(0.8)
	assert filter.prob_in_range(masses, 22., math.inf) == pytest.approx(0.1)
	# bounds which are not bin edges exclude the straddling bins
	assert filter.prob_in_range(masses, 2.3, 5.0) == pytest.approx(0.5)
	assert filter.prob_in_range(masses, 2.5, 5.5) == pytest.approx(0.3)
	assert filter.prob_in_range(masses, 50., math.inf) == pytest.approx(0.0)

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
	cache = RecentEventCache()
	filter = lvk_filter(cache=cache)
	
	result, result_data = filter.should_follow_up(lvk_alert(75.), None)
	assert result
	assert result_data["type"] == "GW_case_Gold"
	# passing BNS/NSBH events are recorded for coincidence checks
	assert len(cache.entries) == 1
	
	result, result_data = filter.should_follow_up(lvk_alert(400.), None)
	assert result
	assert result_data["type"] == "GW_case_Silver"
	
	for area in (800., 1200.):
		result, result_data = filter.should_follow_up(lvk_alert(area), None)
		assert result
		assert result_data["type"] == "GW_case_Bronze"
	
	# Omega_obs larger than 1500 deg^2 is trimmed, and remains Bronze
	result, result_data = filter.should_follow_up(lvk_alert(2500.), None)
	assert result
	assert result_data["type"] == "GW_case_Bronze"
	assert numpy.count_nonzero(result_data["reward_mask"])*PROB_MAP_PIXEL_AREA_DEG2 <= 1500.
	
	# step 1 is passed by any one of its conditions
	result, result_data = filter.should_follow_up(
		lvk_alert(75., classification={"BNS": 0.05, "NSBH": 0.9, "BBH": 0.04, "Noise": 0.01},
		          properties=NO_NS_PROPERTIES), None)
	assert result
	result, result_data = filter.should_follow_up(
		lvk_alert(75., classification=NO_NS_CLASSIFICATION,
		          properties={"HasNS": 0.5, "HasRemnant": 0.0, "HasMassGap": 0.0}), None)
	assert result
	result, result_data = filter.should_follow_up(
		lvk_alert(75., classification=NO_NS_CLASSIFICATION,
		          properties={"HasNS": 0.25, "HasRemnant": 0.10, "HasMassGap": 0.0}), None)
	assert result
	
	# fails all step 1 conditions
	low_ns_prob = lvk_alert(75., classification={"BNS": 0.05, "NSBH": 0.2, "BBH": 0.74, "Noise": 0.01},
	                        properties={"HasNS": 0.25, "HasRemnant": 0.05, "HasMassGap": 0.08,
	                                    "HasSSM": 0.3})
	result, result_data = filter.should_follow_up(low_ns_prob, None)
	assert not result
	
	# P(BNS)+P(NSBH) must be strictly greater than 0.9
	result, result_data = filter.should_follow_up(
		lvk_alert(75., classification={"BNS": 0.5, "NSBH": 0.4, "BBH": 0.1, "Noise": 0.0},
		          properties=NO_NS_PROPERTIES), None)
	assert not result
	
	# false alarm rate
	result, result_data = filter.should_follow_up(lvk_alert(75., far=0.99*FAR_1_PER_YR), None)
	assert result
	result, result_data = filter.should_follow_up(lvk_alert(75., far=4e-8), None)
	assert not result
	
	# localised on the Galactic plane
	result, result_data = filter.should_follow_up(lvk_alert(100., pos=PLANE_POS), None)
	assert not result
	
	# localised near the Sun
	result, result_data = filter.should_follow_up(lvk_alert(100., pos=SUN_POS), None)
	assert not result

def test_LVK_should_follow_up_bns_nsbh_chirp_mass(monkeypatch):
	# P(Mchirp < 2.3) >= 0.9 suffices alone
	filter = lvk_filter(monkeypatch, mass_data({1.1: 0.5, 1.2: 0.45, 2.3: 0.05}))
	result, result_data = filter.should_follow_up(
		lvk_alert(75., classification=NO_NS_CLASSIFICATION, properties=NO_NS_PROPERTIES), None)
	assert result
	assert result_data["type"] == "GW_case_Gold"
	
	filter = lvk_filter(monkeypatch, mass_data({1.1: 0.5, 1.2: 0.39, 2.3: 0.11}))
	result, result_data = filter.should_follow_up(
		lvk_alert(75., classification=NO_NS_CLASSIFICATION, properties=NO_NS_PROPERTIES), None)
	assert not result
	
	# P(2.3 < Mchirp < 5.5) >= 0.9 requires P(BNS)+P(NSBH) > 0.1
	filter = lvk_filter(monkeypatch, mass_data({2.3: 0.6, 3.0: 0.35, 5.5: 0.05}))
	result, result_data = filter.should_follow_up(
		lvk_alert(75., classification={"BNS": 0.0, "NSBH": 0.2, "BBH": 0.8, "Noise": 0.0},
		          properties=NO_NS_PROPERTIES), None)
	assert result
	assert result_data["type"] == "GW_case_Gold"
	result, result_data = filter.should_follow_up(
		lvk_alert(75., classification={"BNS": 0.0, "NSBH": 0.05, "BBH": 0.95, "Noise": 0.0},
		          properties=NO_NS_PROPERTIES), None)
	# (this does, however, pass the chirp mass alternative for lensed BNS mergers)
	assert not any(t.startswith("GW_case_") for t in result_data["passed_types"])

def test_LVK_should_follow_up_lensed_bns_merger(monkeypatch):
	filter = lvk_filter()
	lensed_classification = {"BNS": 0.85, "NSBH": 0.05, "BBH": 0.04, "Noise": 0.01}
	lensed_properties = {"HasNS": 0.9, "HasRemnant": 0.5, "HasMassGap": 0.95, "HasSSM": 0.1}
	
	result, result_data = filter.should_follow_up(
		lvk_alert(800., classification=lensed_classification, properties=lensed_properties), None)
	assert result
	assert result_data["type"] == "lensed_BNS_case_A"
	# lensed BNS takes precedence over BNS/NSBH, but both are recorded
	assert "GW_case_Bronze" in result_data["passed_types"]
	
	result, result_data = filter.should_follow_up(
		lvk_alert(12., classification=lensed_classification, properties=lensed_properties), None)
	assert result
	assert result_data["type"] == "lensed_BNS_case_B"
	
	# too little mass gap probability for lensing, but still a BNS
	result, result_data = filter.should_follow_up(
		lvk_alert(800., classification=lensed_classification,
		          properties=dict(lensed_properties, HasMassGap=0.85)), None)
	assert result
	assert result_data["type"] == "GW_case_Bronze"
	
	# too much NSBH probability for lensing
	result, result_data = filter.should_follow_up(
		lvk_alert(800., classification={"BNS": 0.75, "NSBH": 0.15, "BBH": 0.04, "Noise": 0.01},
		          properties=lensed_properties), None)
	assert "lensed_BNS_case_A" not in result_data["passed_types"]
	
	# false alarm rate too high for any category
	result, result_data = filter.should_follow_up(
		lvk_alert(800., far=4e-8, classification=lensed_classification,
		          properties=lensed_properties), None)
	assert not result
	
	# too poorly localised for lensing
	result, result_data = filter.should_follow_up(
		lvk_alert(1000., classification=lensed_classification, properties=lensed_properties), None)
	assert "lensed_BNS_case_A" not in result_data["passed_types"]
	
	# chirp mass condition can substitute for HasMassGap
	filter = lvk_filter(monkeypatch, mass_data({2.3: 0.6, 3.0: 0.35, 5.5: 0.05}))
	result, result_data = filter.should_follow_up(
		lvk_alert(800., classification=lensed_classification,
		          properties=dict(lensed_properties, HasMassGap=0.0)), None)
	assert result
	assert result_data["type"] == "lensed_BNS_case_A"

def test_LVK_should_follow_up_bbh_merger(monkeypatch):
	heavy = mass_data({22.0: 0.6, 44.0: 0.35, 11.0: 0.05})
	filter = lvk_filter(monkeypatch, heavy)
	
	result, result_data = filter.should_follow_up(
		lvk_alert(90., classification=NO_NS_CLASSIFICATION, properties=NO_NS_PROPERTIES), None)
	assert result
	assert result_data["type"] == "BBH"
	
	# full 90% area must be < 100 deg^2
	result, result_data = filter.should_follow_up(
		lvk_alert(110., classification=NO_NS_CLASSIFICATION, properties=NO_NS_PROPERTIES), None)
	assert not result
	
	# false alarm rate
	result, result_data = filter.should_follow_up(
		lvk_alert(90., far=4e-8, classification=NO_NS_CLASSIFICATION, properties=NO_NS_PROPERTIES),
		None)
	assert not result
	
	# not observable
	result, result_data = filter.should_follow_up(
		lvk_alert(90., pos=SUN_POS, classification=NO_NS_CLASSIFICATION,
		          properties=NO_NS_PROPERTIES), None)
	assert not result
	
	# not heavy enough
	filter = lvk_filter(monkeypatch, mass_data({22.0: 0.5, 44.0: 0.35, 11.0: 0.15}))
	result, result_data = filter.should_follow_up(
		lvk_alert(90., classification=NO_NS_CLASSIFICATION, properties=NO_NS_PROPERTIES), None)
	assert not result

SSM_PROPERTIES = {"HasSSM": 1.0, "HasNS": 0.08, "HasMassGap": 0.0}

def test_LVK_should_follow_up_ssm(monkeypatch):
	# Real SSM alerts have empty classifications and limited properties
	filter = lvk_filter()
	result, result_data = filter.should_follow_up(
		lvk_alert(50., far=5e-9, search="SSM", classification={}, properties=SSM_PROPERTIES), None)
	assert result
	assert result_data["type"] == "SSM_Gold"
	
	result, result_data = filter.should_follow_up(
		lvk_alert(300., far=5e-9, search="SSM", classification={}, properties=SSM_PROPERTIES), None)
	assert result
	assert result_data["type"] == "SSM_Silver"
	
	# larger regions are trimmed to 500 deg^2, and are Silver
	result, result_data = filter.should_follow_up(
		lvk_alert(800., far=5e-9, search="SSM", classification={}, properties=SSM_PROPERTIES), None)
	assert result
	assert result_data["type"] == "SSM_Silver"
	assert numpy.count_nonzero(result_data["reward_mask"])*PROB_MAP_PIXEL_AREA_DEG2 <= 500.
	
	# HasSSM must be at least 0.5
	result, result_data = filter.should_follow_up(
		lvk_alert(50., far=5e-9, search="SSM", classification={},
		          properties=dict(SSM_PROPERTIES, HasSSM=0.5)), None)
	assert result
	result, result_data = filter.should_follow_up(
		lvk_alert(50., far=5e-9, search="SSM", classification={},
		          properties=dict(SSM_PROPERTIES, HasSSM=0.49)), None)
	assert not result
	
	# must come from the SSM search
	result, result_data = filter.should_follow_up(
		lvk_alert(50., far=5e-9, search="AllSky", classification={}, properties=SSM_PROPERTIES), None)
	assert not result
	
	# false alarm rate must be below 1 per 3 years
	result, result_data = filter.should_follow_up(
		lvk_alert(50., far=0.5*(FAR_1_PER_3YR + FAR_1_PER_YR), search="SSM", classification={},
		          properties=SSM_PROPERTIES), None)
	assert not result
	
	# Galactic plane
	result, result_data = filter.should_follow_up(
		lvk_alert(50., pos=PLANE_POS, far=5e-9, search="SSM", classification={},
		          properties=SSM_PROPERTIES), None)
	assert not result
	
	# chirp mass cannot substitute for HasSSM, and SSM events are not treated as BNS/NSBH even when
	# their chirp mass would pass the BNS/NSBH criteria
	filter = lvk_filter(monkeypatch, mass_data({0.1: 0.95, 0.87: 0.05}))
	result, result_data = filter.should_follow_up(
		lvk_alert(50., far=5e-9, search="SSM", classification={},
		          properties=dict(SSM_PROPERTIES, HasSSM=0.0)), None)
	assert not result
	result, result_data = filter.should_follow_up(
		lvk_alert(50., far=5e-9, search="SSM", classification={}, properties=SSM_PROPERTIES), None)
	assert result_data["passed_types"] == ["SSM_Gold"]

def test_LVK_generate_scheduling_data():
	filter = lvk_filter()
	alert = lvk_alert(75.)
	result, result_data = filter.should_follow_up(alert, None)
	assert result
	data = filter.generate_scheduling_data(alert, None, result_data)
	assert data["alert_type"] == "GW_case_Gold"
	assert data["reward_map_nside"] == 32
	assert len(data["reward_map"]) == 12*32*32
	assert data["reward_map"].dtype == bool
	# the order 5 reward map covers the trimmed region
	assert 75. <= numpy.count_nonzero(data["reward_map"])*4*math.pi/(12*32*32)/DEG2_TO_SR < 200.

def make_neutrino_map(area_deg2: float, pos=HIGH_B_POS):
	"""Make a gzipped, flat, RING ordered order 6 map in the style of IceCube"""
	order = 6
	nside = 1<<order
	pixel_area_deg2 = math.pi/(3<<(order<<1))/DEG2_TO_SR
	n_in = max(1, math.ceil(area_deg2/pixel_area_deg2))
	in_pixels, npix = _nearest_pixels(order, n_in, *pos)
	prob = numpy.full(npix, 0.05/(npix - n_in))
	prob[in_pixels] = 0.95/n_in
	ring_prob = numpy.zeros(npix)
	ring_prob[forward_alerts.healpy.nest2ring(nside, numpy.arange(npix))] = prob
	t = astropy.table.Table()
	t["PROB"] = ring_prob
	t.meta["ORDERING"] = "RING"
	buffer = BytesIO()
	t.write(buffer, format="fits")
	return gzip.compress(buffer.getvalue())

def neutrino_alert(p_astro: float=0.6, energy: float=150., pos=HIGH_B_POS, alert_type="initial",
                   trigger_time=T0):
	return {
		"mission": "IceCube",
		"event_name": ["IceCube-261115A"],
		"alert_datetime": trigger_time,
		"alert_type": alert_type,
		"alert_tense": "current",
		"ra": pos[0],
		"dec": pos[1],
		"systematic_included": False,
		"healpix_url": "https://example.com/map.fits.gz",
		"trigger_time": trigger_time,
		"nu_energy": energy,
		"p_astro": p_astro,
	}

@pytest.fixture
def neutrino_map(monkeypatch):
	maps = {"area": 3., "pos": HIGH_B_POS}
	monkeypatch.setattr(forward_alerts, "fetch_file",
	                    lambda url, use_cache=False: make_neutrino_map(maps["area"], maps["pos"]))
	return maps

def test_IceCube_should_follow_up(neutrino_map):
	filter = IceCubeAlertFilter({}, None, True, event_cache=RecentEventCache())
	assert filter.allowed_alert_type == "initial"
	
	# initial alerts are accepted without systematics
	result, result_data = filter.should_follow_up(neutrino_alert(), None)
	assert result
	assert result_data["type"] == "neutrino"
	data = filter.generate_scheduling_data(neutrino_alert(), None, result_data)
	assert data["alert_type"] == "neutrino"
	assert numpy.any(data["reward_map"])
	
	# other alert types are not processed by default
	result, result_data = filter.should_follow_up(neutrino_alert(alert_type="update"), None)
	assert not result
	
	# retracted events
	filter.history["IceCube-261115A"] = {"type": "retraction"}
	result, result_data = filter.should_follow_up(neutrino_alert(), None)
	assert not result
	filter.history.clear()
	
	# Galactic plane
	result, result_data = filter.should_follow_up(neutrino_alert(pos=PLANE_POS), None)
	assert not result
	
	# centroid not observable
	neutrino_map["pos"] = SUN_POS
	result, result_data = filter.should_follow_up(neutrino_alert(pos=SUN_POS), None)
	assert not result
	neutrino_map["pos"] = HIGH_B_POS
	
	# localisation too large for a single pointing
	neutrino_map["area"] = 20.
	result, result_data = filter.should_follow_up(neutrino_alert(), None)
	assert not result

def test_IceCube_should_follow_up_tiers(neutrino_map):
	filter = IceCubeAlertFilter({}, None, True, event_cache=RecentEventCache())
	
	def classify(**kwargs):
		result, result_data = filter.should_follow_up(neutrino_alert(**kwargs), None)
		return result_data["type"] if result else None
	
	# EHE requires both energy > 1 PeV and p_astro > 0.5
	assert classify(p_astro=0.6, energy=1500.) == "neutrino_EHE"
	assert classify(p_astro=0.45, energy=1500.) == "neutrino"
	assert classify(p_astro=0.5, energy=1500.) == "neutrino"
	assert classify(p_astro=0.6, energy=1000.) == "neutrino"
	# standard requires p_astro > 0.4
	assert classify(p_astro=0.41) == "neutrino"
	assert classify(p_astro=0.40) is None
	assert classify(p_astro=0.35) is None

def test_IceCube_should_follow_up_coincident(neutrino_map):
	cache = RecentEventCache()
	gw_skymap = skymap_from_bytes(make_skymap_at(100., *HIGH_B_POS))
	cache.add(forward_alerts.LVK_BNS_NSBH_EVENT, "S261115a",
	          Time(T0) + 5*u.min, gw_skymap.credible_mask(0.9))
	
	# disabled by default
	filter = IceCubeAlertFilter({}, None, True, event_cache=cache)
	result, result_data = filter.should_follow_up(neutrino_alert(p_astro=0.35), None)
	assert not result
	
	filter = IceCubeAlertFilter({}, None, True, enable_coincidence=True, event_cache=cache)
	result, result_data = filter.should_follow_up(neutrino_alert(p_astro=0.35), None)
	assert result
	assert result_data["type"] == "neutrino_coincident"
	
	# EHE takes precedence
	result, result_data = filter.should_follow_up(neutrino_alert(p_astro=0.6, energy=2000.), None)
	assert result_data["type"] == "neutrino_EHE"
	
	# too far apart in time
	late = (Time(T0) + 20*u.min).isot + "Z"
	result, result_data = filter.should_follow_up(neutrino_alert(p_astro=0.35, trigger_time=late), None)
	assert not result
	
	# no spatial overlap
	far_pos = (HIGH_B_POS[0] + 40., HIGH_B_POS[1])
	neutrino_map["pos"] = far_pos
	result, result_data = filter.should_follow_up(neutrino_alert(p_astro=0.35, pos=far_pos), None)
	assert not result

def test_coincidence_from_LVK_filter(neutrino_map):
	cache = RecentEventCache()
	gw_filter = lvk_filter(cache=cache)
	result, result_data = gw_filter.should_follow_up(lvk_alert(75.), None)
	assert result
	nu_filter = IceCubeAlertFilter({}, None, True, enable_coincidence=True, event_cache=cache)
	result, result_data = nu_filter.should_follow_up(neutrino_alert(p_astro=0.35), None)
	assert result
	assert result_data["type"] == "neutrino_coincident"

def test_RecentEventCache_max_age():
	cache = RecentEventCache()
	assert cache.max_age_s == 6*3600
	mask = numpy.zeros(12<<(2*PROB_MAP_ORDER), dtype=bool)
	cache.add("A", "first", T0, mask)
	cache.add("A", "second", Time(T0) + 5*u.hour, mask)
	assert cache.get("A", "first") is not None
	cache.add("A", "third", Time(T0) + 7*u.hour, mask)
	assert cache.get("A", "first") is None
	assert cache.get("A", "second") is not None
	# re-adding the same source replaces its entry
	cache.add("A", "third", Time(T0) + 7*u.hour, mask, extra=1)
	assert len([e for e in cache.entries if e["source"] == "third"]) == 1
	assert cache.get("A", "third")["extra"] == 1
	cache.remove("A", "third")
	assert cache.get("A", "third") is None

class CaptureSender:
	def __init__(self):
		self.sent = []
	
	def send(self, data: dict, test: bool=False):
		self.sent.append(data)

def coincidence_filters(enable_coincidence: bool=True):
	"""LVK and IceCube filters sharing a history, sender, and event cache, as in the main program"""
	history = {}
	sender = CaptureSender()
	cache = RecentEventCache()
	gw_filter = LVKAlertFilter(history, sender, True, event_cache=cache)
	nu_filter = IceCubeAlertFilter(history, sender, True, enable_coincidence=enable_coincidence,
	                               event_cache=cache)
	return gw_filter, nu_filter, sender

def test_late_LVK_alert_upgrades_neutrino(neutrino_map):
	# a neutrino too weak for a standard alert is remembered, and followed up once a coincident
	# BNS/NSBH alert passes
	gw_filter, nu_filter, sender = coincidence_filters()
	assert not nu_filter.process(neutrino_alert(p_astro=0.35), None)
	assert len(sender.sent) == 0
	assert gw_filter.process(lvk_alert(75.), None)
	assert [d["alert_type"] for d in sender.sent] == ["GW_case_Gold", "neutrino_coincident"]
	upgrade = sender.sent[1]
	assert upgrade["source"] == "IceCube-261115A"
	assert not upgrade["is_update"]
	assert upgrade["instrument"][0] == "IceCube"
	assert upgrade["event_trigger_timestamp"] == T0
	assert numpy.any(upgrade["reward_map"])
	
	# a neutrino which already had a standard alert gets an update
	gw_filter, nu_filter, sender = coincidence_filters()
	assert nu_filter.process(neutrino_alert(p_astro=0.6), None)
	assert gw_filter.process(lvk_alert(75.), None)
	assert [d["alert_type"] for d in sender.sent] == ["neutrino", "GW_case_Gold", "neutrino_coincident"]
	assert sender.sent[2]["is_update"]
	# repeated LVK alerts do not cause repeated upgrades
	assert not gw_filter.process(lvk_alert(75.), None)
	assert len(sender.sent) == 3
	
	# EHE neutrinos are not changed
	gw_filter, nu_filter, sender = coincidence_filters()
	assert nu_filter.process(neutrino_alert(p_astro=0.6, energy=2000.), None)
	assert gw_filter.process(lvk_alert(75.), None)
	assert [d["alert_type"] for d in sender.sent] == ["neutrino_EHE", "GW_case_Gold"]
	
	# when the LVK alert is processed first, the neutrino is coincident when it arrives, and is not
	# upgraded again
	gw_filter, nu_filter, sender = coincidence_filters()
	assert gw_filter.process(lvk_alert(75.), None)
	assert nu_filter.process(neutrino_alert(p_astro=0.35), None)
	assert gw_filter.process(lvk_alert(75.), None) is False
	assert [d["alert_type"] for d in sender.sent] == ["GW_case_Gold", "neutrino_coincident"]

def test_late_LVK_alert_no_upgrade(neutrino_map):
	# disabled
	gw_filter, nu_filter, sender = coincidence_filters(enable_coincidence=False)
	nu_filter.process(neutrino_alert(p_astro=0.35), None)
	assert gw_filter.process(lvk_alert(75.), None)
	assert [d["alert_type"] for d in sender.sent] == ["GW_case_Gold"]
	
	# neutrino retracted
	gw_filter, nu_filter, sender = coincidence_filters()
	nu_filter.process(neutrino_alert(p_astro=0.35), None)
	nu_filter.process(neutrino_alert(p_astro=0.35, alert_type="retraction"), None)
	assert gw_filter.process(lvk_alert(75.), None)
	assert [d["alert_type"] for d in sender.sent] == ["GW_case_Gold"]
	
	# too far apart in time
	gw_filter, nu_filter, sender = coincidence_filters()
	early = (Time(T0) - 20*u.min).isot + "Z"
	nu_filter.process(neutrino_alert(p_astro=0.35, trigger_time=early), None)
	assert gw_filter.process(lvk_alert(75.), None)
	assert [d["alert_type"] for d in sender.sent] == ["GW_case_Gold"]
	
	# no spatial overlap
	gw_filter, nu_filter, sender = coincidence_filters()
	far_pos = (HIGH_B_POS[0] + 40., HIGH_B_POS[1])
	neutrino_map["pos"] = far_pos
	nu_filter.process(neutrino_alert(p_astro=0.35, pos=far_pos), None)
	assert gw_filter.process(lvk_alert(75.), None)
	assert [d["alert_type"] for d in sender.sent] == ["GW_case_Gold"]
	
	# the LVK alert does not pass as a BNS/NSBH merger
	gw_filter, nu_filter, sender = coincidence_filters()
	neutrino_map["pos"] = HIGH_B_POS
	nu_filter.process(neutrino_alert(p_astro=0.35), None)
	gw_filter.process(lvk_alert(75., far=4e-8), None)
	assert len(sender.sent) == 0
