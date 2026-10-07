#!/usr/bin/env python3

import argparse
from astropy.coordinates import AltAz, EarthLocation, Galactic, ICRS, SkyCoord, TETE, get_sun
from astropy.io import fits
import astropy.table
from astropy.time import Time
import astropy.units
from astropy.utils import iers
from astropy_healpix import HEALPix, boundaries_lonlat, healpy
from astropy_healpix.core import healpix_cone_search
import copy
from dataclasses import dataclass
import datetime
import fastavro
import functools
import gzip
import hashlib
import hop
from io import BytesIO
import json
import logging
import math
import numpy
import orjson
import os
import requests
import sys
from typing import Mapping, Sequence
from urllib.parse import urlparse
import yaml
import zlib

logger = logging.getLogger("ToO Alert Producer")
use_file_cache = False
treat_cache_miss_as_not_found = False

# Trigger thresholds. Section numbers refer to the "Rubin ToO 2026 revised strategy" report.

DEG2_TO_SR = (math.pi/180.)**2
# False alarm rate of one per year (~3.17e-8 Hz). The report quotes "1.6e-8 Hz", which is a typo for
# 1/yr and is deliberately not used.
FAR_1_PER_YR = 1./(365.25*86400.)
# False alarm rate of one per three years (~1.06e-8 Hz), used for sub-solar mass candidates (§2.3.3)
FAR_1_PER_3YR = FAR_1_PER_YR/3.

# Observability (§2.1.3 step 3, also used by §2.2.3, §2.3.3 and §3)
RUBIN_LAT_DEG = -30.2446
RUBIN_LON_DEG = -70.7494
RUBIN_HEIGHT_M = 2663.
OBS_WINDOW_H = 24.          # look for observability within 24 hours of the event
OBS_SUN_ALT_MAX_DEG = -12.  # Sun must be below nautical twilight
OBS_MIN_ALT_DEG = 30.       # airmass < 2
OBS_TIME_STEP_MIN = 10.     # sampling interval within the observability window
# HEALPix order used for per-pixel probability and observability work (pixels of ~0.21 deg^2)
PROB_MAP_ORDER = 7
PROB_MAP_PIXEL_AREA_DEG2 = (4*math.pi/(12<<(PROB_MAP_ORDER<<1)))/DEG2_TO_SR
# Standard visibility pre-cut: a credible region lying entirely north of this declination never
# reaches airmass < 2 from Rubin
MIN_DEC_CUT_DEG = 30.
GAL_LAT_CUT_DEG = 10.       # |b| > 10 deg (§2.1.3 step 4, §2.3.3, §3)
GW_CRED = 0.9               # credible level used to define the GW localisation region

# BNS and NSBH mergers (§2.1.3)
BNS_NSBH_CLASS_MIN = 0.9      # step 1: P(BNS)+P(NSBH) > 0.9
HASNS_MIN = 0.5               # step 1: HasNS >= 0.5
HASREMNANT_MIN = 0.10         # step 1: HasRemnant >= 0.10
MCHIRP_NS_MAX = 2.3           # step 1: P(Mchirp < 2.3 Msun) >= 0.9
MCHIRP_GAP_MAX = 5.5          # step 1: P(2.3 < Mchirp < 5.5 Msun) >= 0.9 ...
BNS_NSBH_GAP_CLASS_MIN = 0.1  #         ... and P(BNS)+P(NSBH) > 0.1
MCHIRP_PROB_MIN = 0.9         # probability required for all chirp-mass range conditions
BNS_MAX_AREA_DEG2 = 1500.     # step 3: trim Omega_obs to at most this area
HIGH_B_PROB_MIN = 0.10        # step 4: probability in Omega_obs with |b| > 10 deg (also §2.3.3)
BNS_GOLD_DEG2 = 100.          # step 5: Gold < 100 deg^2
BNS_SILVER_DEG2 = 500.        # step 5: Silver < 500 deg^2, Bronze < 1500 deg^2

# BBH mergers (§2.2.3)
BBH_MCHIRP_MIN = 22.          # P(Mchirp >= 22 Msun) >= 0.9
BBH_MAX_AREA_DEG2 = 100.      # full 90% credible area < 100 deg^2
BBH_OBS_FRAC_MIN = 0.70       # fraction of total probability in observable pixels > 0.70

# Sub-solar mass candidates (§2.3.3)
SSM_SEARCH = "SSM"            # value of event.search for the SSM search pipeline(s)
HASSSM_MIN = 0.5              # HasSSM >= 0.5
MCHIRP_SSM_MAX = 0.87         # P(Mchirp < 0.87 Msun) >= 0.9
SSM_MAX_AREA_DEG2 = 500.      # Omega_obs above this is Bronze-level, which is not automated
SSM_GOLD_DEG2 = 100.          # Gold < 100 deg^2, Silver < 500 deg^2

# Gravitationally lensed BNS mergers (§5.1.3)
LENSED_MASSGAP_MIN = 0.9      # HasMassGap >= 0.9 ...
LENSED_MCHIRP_MIN = 2.3       # ... or P(2.3 < Mchirp < 5.5 Msun) >= 0.9
# The report says 5 Msun, but the LVK chirp-mass bins have no edge there (edges are 3.0 and 5.5),
# so the nearest edge above is used.
LENSED_MCHIRP_MAX = 5.5
LENSED_NSBH_MAX = 0.1         # P(NSBH) < 0.1
LENSED_MAX_DEG2 = 900.        # 90% credible area <= 900 deg^2
LENSED_GOLD_DEG2 = 15.        # Gold < 15 deg^2

# Neutrinos (§3)
NU_CRED = 0.9                 # >= 90% of the contour within a single pointing
RUBIN_FOV_SR = 9.6*DEG2_TO_SR # Rubin field of view (~0.002924 sr)
NU_EHE_ENERGY_TEV = 1000.     # EHE: energy > 1 PeV (nu_energy is in TeV) ...
NU_EHE_PASTRO = 0.5           # ... and p_astro > 0.5
NU_COINC_PASTRO = 0.3         # coincident with LVK BNS/NSBH: p_astro > 0.3 ...
NU_COINC_DT_S = 600.          # ... and |dT| < 10 minutes
NU_STD_PASTRO = 0.4           # standard: p_astro > 0.4
# How long events are kept for coincidence checks. LVK Initial alerts typically arrive well after
# neutrino alerts for the same time, so neutrino alerts must be kept long enough to be matched.
COINCIDENCE_CACHE_MAX_AGE_S = 6*3600.

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
		y = float(value) - self.comp
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
		# caches of fixed-order maps derived from the data, keyed by order or (credible level, order)
		self._flat_prob_maps = {}
		self._credible_masks = {}
	
	def area_for_probability(self, target_probability: float):
		"""
		Compute the area on the sky subtended by the highest density portion of the map which
		sums to the target probability.
		
		Return: The sky area, and the minimum declination, in radians, touched by that area
		"""
		summed_prob=KahanAdder()
		summed_area=KahanAdder()
		n_indices = {}
		for p_dens, u_idx in self.data:
			order=math.floor(math.log2(u_idx/4)/2)
			area = self.pixel_areas[order]
			prob=p_dens * area
			summed_prob+=prob
			summed_area+=area
			n_idx = u_idx - (4<<(2*order))
			if 1<<order not in n_indices:
				n_indices[1<<order] = [n_idx]
			else:
				n_indices[1<<order].append(n_idx)
			if summed_prob >= target_probability:
				break
		# Each call to boundaries_lonlat and fixing its output to be a usable form is very slow.
		# This can be worked around by having it process an array of pixels, b ut it cannot handle
		# more than one value of 'nside' at a time, so it must be called once for each order of
		# pixels to be checked.
		min_dec = 100
		for nside in n_indices.keys():
			# extract the declinations (latitudes) of all pixel corners, in radians
			corner_decs = boundaries_lonlat(n_indices[nside], 1, nside, "nested")[1].to_value().flatten()
			min_corner_dec = numpy.min(corner_decs)
			if min_corner_dec < min_dec:
				min_dec = min_corner_dec
		return summed_area.sum, min_dec
	
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
	
	def flat_prob_map(self, order: int = PROB_MAP_ORDER):
		"""
		Make a single-order, NESTED map of the probability (not probability density) in each pixel.
		Pixels of the original map coarser than the target order spread their probability evenly
		over their children; finer pixels add their probability to their parent.
		The result is cached, and must not be modified by the caller.
		"""
		if order in self._flat_prob_maps:
			return self._flat_prob_maps[order]
		u_indices = self.data["uniq_index"]
		densities = self.data["prob_density"]
		orders = numpy.floor(numpy.log2(u_indices)/2).astype(int) - 1
		n_indices = u_indices - numpy.left_shift(4, 2*orders)
		target_area = math.pi/(3<<(order<<1))
		flat_map = numpy.zeros(12<<(order<<1))
		for pix_order in numpy.unique(orders):
			sel = orders == pix_order
			if pix_order <= order:
				# each unit of increase in order adds two bits to the child pixel indices
				shift = 2*(order - pix_order)
				children = (n_indices[sel, numpy.newaxis] << shift) + numpy.arange(1<<shift)
				flat_map[children] = densities[sel, numpy.newaxis] * target_area
			else:
				# throw away two bits for each unit of difference in order to find parent pixel
				parents = n_indices[sel] >> (2*(pix_order - order))
				numpy.add.at(flat_map, parents, densities[sel] * math.pi/(3<<(int(pix_order)<<1)))
		flat_map.flags.writeable = False
		self._flat_prob_maps[order] = flat_map
		return flat_map
	
	def credible_mask(self, cred: float, order: int = PROB_MAP_ORDER):
		"""
		Make a single-order, NESTED boolean map of the highest-probability pixels whose summed
		probability first reaches cred. The result is cached, and must not be modified by the caller.
		"""
		if (cred, order) in self._credible_masks:
			return self._credible_masks[(cred, order)]
		prob = self.flat_prob_map(order)
		ranked = numpy.argsort(prob, kind="stable")[::-1]
		n_pix = min(int(numpy.searchsorted(numpy.cumsum(prob[ranked]), cred)) + 1, len(prob))
		mask = numpy.zeros(len(prob), dtype=bool)
		mask[ranked[:n_pix]] = True
		mask.flags.writeable = False
		self._credible_masks[(cred, order)] = mask
		return mask
	
	def make_flat_binary_map(self, target_probability: float, target_order = None):
		# first, figure out how many pixels it must be from its order, which must be the maximum
		# order in the original map
		if target_order is None:
			target_order=max(self.pixel_areas.keys())
		flat_n_npixels=12<<(target_order<<1)
		#print(f"Making order {target_order} map with {flat_n_npixels} pixels")
		
		flat_map = numpy.zeros(flat_n_npixels, dtype=bool)
		
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
				flat_map[min_idx:(min_idx+idx_range)] = True
			else:
				# throw away two bits for each unit of difference in order to find parent pixel
				p_idx = n_idx >> (2 * (order - target_order))
				flat_map[p_idx] = True
			
			area = self.pixel_areas[order]
			prob=p_dens * area
			summed_prob+=prob
			
			if summed_prob >= target_probability:
				break
		
		return flat_map


RUBIN_LOCATION = EarthLocation(lat=RUBIN_LAT_DEG*astropy.units.deg,
                               lon=RUBIN_LON_DEG*astropy.units.deg,
                               height=RUBIN_HEIGHT_M*astropy.units.m)

def _observable(ra_deg, dec_deg, t0, window_h: float=OBS_WINDOW_H,
                sun_alt_max: float=OBS_SUN_ALT_MAX_DEG, min_alt: float=OBS_MIN_ALT_DEG):
	"""
	Determine whether sky positions are observable from Rubin: a position is observable if at any
	sample time (every OBS_TIME_STEP_MIN minutes) in [t0, t0+window_h] the Sun is below sun_alt_max
	and the position is above min_alt.
	
	Target altitudes are computed from the apparent (TETE) coordinates at t0 and the apparent local
	sidereal time, ignoring refraction and the drift of apparent coordinates within the window; this
	is accurate to far better than needed for an altitude cut.
	
	Args:
	    ra_deg, dec_deg: ICRS coordinates, in degrees (scalars or arrays)
	    t0: Start of the window, in any form accepted by astropy.time.Time
	    window_h: Length of the window, in hours
	    sun_alt_max: Maximum altitude of the Sun, in degrees
	    min_alt: Minimum target altitude, in degrees
	Return: A boolean array with the shape of ra_deg
	"""
	ra = numpy.atleast_1d(numpy.asarray(ra_deg, dtype=float))
	dec = numpy.atleast_1d(numpy.asarray(dec_deg, dtype=float))
	observable = numpy.zeros(ra.shape, dtype=bool)
	# Earth orientation data only affect positions at the sub-arcsecond level, which is irrelevant
	# here, so do not fail when only stale or predicted IERS data are available.
	with iers.conf.set_temp("auto_max_age", None), \
	     iers.conf.set_temp("iers_degraded_accuracy", "ignore"):
		t0 = Time(t0)
		n_steps = int(math.floor(window_h*60./OBS_TIME_STEP_MIN)) + 1
		times = t0 + numpy.arange(n_steps)*OBS_TIME_STEP_MIN*astropy.units.min
		sun_alt = get_sun(times).transform_to(AltAz(obstime=times, location=RUBIN_LOCATION)).alt.deg
		night = times[sun_alt < sun_alt_max]
		if len(night) == 0:
			return observable
		apparent = ICRS(ra=ra*astropy.units.deg, dec=dec*astropy.units.deg).transform_to(TETE(obstime=t0))
		lsts = night.sidereal_time("apparent", longitude=RUBIN_LOCATION.lon).rad
	app_ra = apparent.ra.rad
	sin_part = numpy.sin(apparent.dec.rad) * math.sin(math.radians(RUBIN_LAT_DEG))
	cos_part = numpy.cos(apparent.dec.rad) * math.cos(math.radians(RUBIN_LAT_DEG))
	sin_min_alt = math.sin(math.radians(min_alt))
	for lst in lsts:
		observable |= (sin_part + cos_part*numpy.cos(lst - app_ra)) > sin_min_alt
	return observable

@functools.lru_cache(maxsize=4)
def _pixel_centres(nside: int):
	"""ICRS coordinates, in degrees, of the centres of all NESTED pixels at nside"""
	centres = HEALPix(nside=nside, order="nested", frame=ICRS()).healpix_to_skycoord(numpy.arange(12*nside*nside))
	return centres.ra.deg, centres.dec.deg

@functools.lru_cache(maxsize=4)
def high_galactic_latitude_mask(nside: int):
	"""A NESTED boolean map of pixels whose centres have |b| > GAL_LAT_CUT_DEG"""
	ra, dec = _pixel_centres(nside)
	b = ICRS(ra=ra*astropy.units.deg, dec=dec*astropy.units.deg).transform_to(Galactic()).b.deg
	mask = numpy.abs(b) > GAL_LAT_CUT_DEG
	mask.flags.writeable = False
	return mask

@functools.lru_cache(maxsize=16)
def _observable_mask(nside: int, t0_iso: str, window_h: float, sun_alt_max: float, min_alt: float):
	ra, dec = _pixel_centres(nside)
	mask = _observable(ra, dec, t0_iso, window_h, sun_alt_max, min_alt)
	mask.flags.writeable = False
	return mask

def observable_mask(nside: int, t0, window_h: float=OBS_WINDOW_H,
                    sun_alt_max: float=OBS_SUN_ALT_MAX_DEG, min_alt: float=OBS_MIN_ALT_DEG):
	"""
	Make a NESTED boolean map of the pixels (judged by their centres) which are observable from Rubin
	at some time in [t0, t0+window_h], with the Sun below sun_alt_max and the pixel above min_alt
	(30 degrees corresponds to airmass 2). See _observable.
	The result is cached, and must not be modified by the caller.
	"""
	return _observable_mask(int(nside), Time(t0).isot, float(window_h), float(sun_alt_max),
	                        float(min_alt))

def observable_credible_region(skymap: Skymap, t0, cred: float=GW_CRED, **obs_settings):
	"""
	Compute Omega_obs, the observable part of the credible region, at order PROB_MAP_ORDER.
	Return: A tuple of the per-pixel probability map and the Omega_obs boolean map
	"""
	prob = skymap.flat_prob_map(PROB_MAP_ORDER)
	obs = observable_mask(1<<PROB_MAP_ORDER, t0, **obs_settings)
	return prob, skymap.credible_mask(cred, PROB_MAP_ORDER) & obs

def trim_to_observable(skymap: Skymap, t0, max_area_deg2: float, cred: float=GW_CRED,
                       **obs_settings):
	"""
	Compute Omega_obs, the observable part of the cred credible region, and if its area is larger
	than max_area_deg2 keep only its highest-probability pixels whose total area is no more than
	max_area_deg2. Observability is evaluated by observable_mask, which takes obs_settings.
	
	Probabilities are NOT renormalised: the probabilities returned are fractions of the total
	probability of the whole map, so they do not account for unobservable or trimmed regions, and
	the mask is a binary map which carries no weighting by probability.
	
	Return: A tuple of the NESTED boolean pixel mask at order PROB_MAP_ORDER, its area in square
	        degrees, the total probability in the mask, and the probability in the part of the mask
	        with |b| > GAL_LAT_CUT_DEG.
	"""
	prob, omega = observable_credible_region(skymap, t0, cred, **obs_settings)
	untrimmed_area = numpy.count_nonzero(omega) * PROB_MAP_PIXEL_AREA_DEG2
	if untrimmed_area > max_area_deg2:
		max_pixels = int(math.floor(max_area_deg2/PROB_MAP_PIXEL_AREA_DEG2))
		candidates = numpy.flatnonzero(omega)
		ranked = candidates[numpy.argsort(prob[candidates], kind="stable")[::-1]]
		mask = numpy.zeros(len(prob), dtype=bool)
		mask[ranked[:max_pixels]] = True
		logger.info(f"    Omega_obs area of {untrimmed_area:.1f} deg² exceeds {max_area_deg2} deg²; "
		            "keeping only the highest-probability pixels")
	else:
		mask = omega.copy()
	area = numpy.count_nonzero(mask) * PROB_MAP_PIXEL_AREA_DEG2
	total_prob = float(prob[mask].sum())
	high_b_prob = float(prob[mask & high_galactic_latitude_mask(1<<PROB_MAP_ORDER)].sum())
	return mask, area, total_prob, high_b_prob

def observable_credible_area(skymap: Skymap, t0, cred: float=GW_CRED, **obs_settings):
	"""The area, in square degrees, of the untrimmed Omega_obs"""
	return numpy.count_nonzero(observable_credible_region(skymap, t0, cred, **obs_settings)[1]) * \
	       PROB_MAP_PIXEL_AREA_DEG2

def downgrade_mask(mask, from_order: int, to_order: int):
	"""Reduce a NESTED boolean map in order, marking each parent pixel if any child is marked"""
	return mask.reshape(-1, 1<<(2*(from_order - to_order))).any(axis=1)


class RecentEventCache:
	"""
	A small store of recently seen events, shared between filters so that one filter can look for
	coincidences with events passed by another. Entries whose event times are more than max_age_s
	older than the newest event added are discarded.
	"""
	def __init__(self, max_age_s: float=COINCIDENCE_CACHE_MAX_AGE_S):
		self.max_age_s = max_age_s
		self.entries = []
	
	def add(self, kind: str, source: str, time, mask, **extra):
		"""
		Add an event, replacing any previous entry of the same kind for the same source.
		
		Args:
		    kind: A label for the category of event
		    source: The event identifier
		    time: The event time, in any form accepted by astropy.time.Time
		    mask: A NESTED boolean map of the event's credible region at order PROB_MAP_ORDER
		    extra: Any additional data to store in the entry
		Return: The new entry
		"""
		time = Time(time)
		self.entries = [e for e in self.entries
		                if not (e["kind"] == kind and e["source"] == source) and
		                   (time - e["time"]).sec <= self.max_age_s]
		entry = dict(extra, kind=kind, source=source, time=time, mask=mask)
		self.entries.append(entry)
		return entry
	
	def get(self, kind: str, source: str):
		"""Return the entry of the given kind for the given source, or None"""
		for e in self.entries:
			if e["kind"] == kind and e["source"] == source:
				return e
		return None
	
	def remove(self, kind: str, source: str):
		self.entries = [e for e in self.entries if not (e["kind"] == kind and e["source"] == source)]
	
	def find(self, kind: str, time, max_dt_s: float):
		"""Return all entries of the given kind with times within max_dt_s seconds of time"""
		time = Time(time)
		return [e for e in self.entries
		        if e["kind"] == kind and abs((e["time"] - time).sec) < max_dt_s]

# The cache used by default by all filters
recent_events = RecentEventCache()

# Category labels for events stored in the recent event cache
LVK_BNS_NSBH_EVENT = "LVK_BNS_NSBH"
NEUTRINO_EVENT = "NEUTRINO"


def write_json(records, compressed: bool=False):
	buf=orjson.dumps(records, option=orjson.OPT_SERIALIZE_NUMPY)
	if(compressed):
		zbuf=zlib.compress(buf, level=9)
		return zbuf
	else:
		return buf


def fetch_file(url: str, use_cache: bool = False):
	if use_cache:
		h = hashlib.sha256()
		h.update(url.encode("utf-8"))
		nh = h.hexdigest()
		try:
			with open(f"file_cache/{nh}", "rb") as f:
				logger.info(f"Reading cached data from {url} from file_cache/{nh}")
				return f.read()
		except FileNotFoundError:
			if treat_cache_miss_as_not_found:
				raise RuntimeError(f"Treating {url} as not found due to lack of cache entry file_cache/{nh}")
			pass
	logger.info(f"Requesting {url}")
	resp = requests.get(url)
	if resp.status_code != 200:
		raise RuntimeError(f"HTTP GET of {url} failed ({resp.status_code}): {resp.content}")
	if use_cache:
		try:
			os.stat("file_cache")
		except FileNotFoundError:
			os.mkdir("file_cache")
		with open(f"file_cache/{nh}", "wb") as f:
			f.write(resp.content)
	return resp.content


# not actually a class, just a wrapper function for working the annoying hop.io.Stream middleman
def KafkaConsumer(url: str, *args, **kwargs):
	stream_kwargs = {}
	stream_arg_names = ["auth", "start_at", "until_eos"]
	for arg_name in stream_arg_names:
		if arg_name in kwargs:
			stream_kwargs[arg_name]=kwargs.pop(arg_name)
	return hop.io.Stream(**stream_kwargs).open(url, mode='r', *args, **kwargs)


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
			fileext = os.path.splitext(filepath)[1]
			format = fileext.upper()[1:]
			if format in hop.io.Deserializer.__members__:
				msg = hop.io.Deserializer[format].load_file(filepath)
			else:
				logging.warning(f"Message format {format} not recognized; returning a Blob")
				msg = hop.models.Blob.load_file(filepath)
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


class FileSender(AlertSender):
	def __init__(self, output_schema, output_dir):
		super().__init__(output_schema)
		self.output_dir = output_dir
	
	def send(self, data: dict, test: bool=False):
		if test:
			print("*** TEST ALERT ***")
		fname = f"{self.output_dir}/{data['source']}.json"
		with open(fname, "wb") as f:
			f.write(write_json(data, compressed=False))


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
		self.schema = json.dumps(self.schema)
		self.url = url
	
	def send(self, data: dict, test: bool=False):
		raw_body = {"value_schema": self.schema,
		            "records": [{"value": data}],
		            }
		request_body = write_json(raw_body, compressed=False)
		
		additional_headers = {"Content-Type": "application/vnd.kafka.avro.v2+json",
		                      #"Content-Encoding": "gzip",
		                      }
		resp = requests.post(self.url, data=request_body, headers=additional_headers)
		if resp.status_code != 200:
			logging.error(f"POST to Confluent REST Proxy failed ({resp.status_code}): "
			              f"{resp.content}")
			# TODO: figure out what if anything else we should do about errors


class AlertFilter:
	last_timestamp = 0

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
		        determine whether another alert with the same identifier supersedes the one(s)
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
		self.send_scheduling_data(scheduling_data, alert_id, is_test, is_update)
		self.after_send(message, metadata, alert_data, alert_id, is_test)
		return True
	
	def send_scheduling_data(self, scheduling_data: dict, alert_id, is_test: bool, is_update: bool):
		"""
		Fill in the common fields of the data for the scheduler, and send it.
		"""
		# Temporary hack: pad or truncate the instrument list to a length of exactly 3
		while len(scheduling_data["instrument"]) < 3:
			scheduling_data["instrument"].append("")
		while len(scheduling_data["instrument"]) > 3:
			scheduling_data["instrument"].pop()
		scheduling_data["source"] = alert_id
		scheduling_data["is_test"] = is_test
		scheduling_data["is_update"] = is_update
		timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
		if timestamp <= AlertFilter.last_timestamp:
			timestamp = AlertFilter.last_timestamp + 1
		AlertFilter.last_timestamp = timestamp
		scheduling_data["timestamp"] = timestamp
		
		self.sender.send(scheduling_data, test=is_test)
	
	def after_send(self, message, metadata, alert_data, alert_id, is_test: bool):
		"""
		Called after data for an alert has been sent to the scheduler, for any further actions.
		The default implementation does nothing.
		"""
		pass


class LVKAlertFilter(AlertFilter):
	def __init__(self, history: dict, sender: AlertSender, allow_tests: bool=False,
	             alert_type="INITIAL", obs_window_h: float=OBS_WINDOW_H,
	             obs_sun_alt_max_deg: float=OBS_SUN_ALT_MAX_DEG,
	             obs_min_alt_deg: float=OBS_MIN_ALT_DEG, event_cache: RecentEventCache=None):
		"""
		Args:
		    obs_window_h, obs_sun_alt_max_deg, obs_min_alt_deg: Observability settings passed to
		        observable_mask.
		    event_cache: Where to record passing BNS/NSBH events for coincidence checks by other
		                 filters. Defaults to the shared module-level cache.
		"""
		super().__init__(history, sender, allow_tests)
		self.allowed_alert_type = alert_type
		if alert_type != "INITIAL":
			logger.warning(f"LVKAlertFilter alert type is {alert_type}, not INITIAL")
		self.obs_settings = {"window_h": obs_window_h, "sun_alt_max": obs_sun_alt_max_deg,
		                     "min_alt": obs_min_alt_deg}
		self.event_cache = event_cache if event_cache is not None else recent_events
	
	def is_test(self, message, metadata):
		if super().is_test(message, metadata):
			return True
		# "Prefix: S for normal candidates and MS or TS for mock or test events, respectively"
		return message["superevent_id"][0] != "S"
	
	def alert_identifier(self, message, metadata):
		if "event" in message and message["event"] is not None:
			t = message["event"]["time"]
		else:
			t = message["time_created"]
		return message["superevent_id"], \
		       {"type": message["alert_type"].upper(), "time": t}
	
	def overrides_previous(self, old_meta, new_meta):
		# Retractions override previous alerts.
		# If there were an interest in handling updates, etc., logic should be added here.
		return old_meta["type"] == "INITIAL" and new_meta["type"] == "RETRACTION"

	def get_chirp_mass_estimate(self, message, metadata):
		"""Return: a tuple of mass bin egdes and bin probabilities, or None if no suitable data
		           could be downloaded.
		"""
		no_mass_data = None
		if "urls" not in message or "gracedb" not in message["urls"]:
			return no_mass_data
		parsed_gracedb_url = urlparse(message["urls"]["gracedb"])
		mass_url = f"https://{parsed_gracedb_url.netloc}/api/superevents/" \
		           f"{self.alert_identifier(message,metadata)[0]}/files/mchirp_source.json"
		try:
			raw_mass_data = fetch_file(mass_url, use_file_cache)
		except Exception as ex:
			logger.warning(f"Failed to fetch mass data from {mass_url}: {ex}")
			return no_mass_data
		try:
			mass_data = json.loads(raw_mass_data)
		except Exception as ex:
			logger.warning(f"Data fetched from {mass_url} cannot be decoded as JSON: {ex}")
			return no_mass_data
		if not isinstance(mass_data, Mapping):
			logger.warning(f"Data fetched from {mass_url} is not a JSON mapping")
			return no_mass_data
		if "bin_edges" not in mass_data or "probabilities" not in mass_data:
			logger.warning(f"Data fetched from {mass_url} does not have both bin_edges and "
			               "probabilities keys")
			return no_mass_data
		if not isinstance(mass_data["bin_edges"], Sequence) or \
		  not isinstance(mass_data["probabilities"], Sequence):
			logger.warning(f"Data fetched from {mass_url} does not have sequences for both "
			               "bin_edges and probabilities")
			return no_mass_data
		if len(mass_data["bin_edges"]) != 1 + len(mass_data["probabilities"]):
			logger.warning(f"Data fetched from {mass_url} does not have compatible lengths for "
			               "bin_edges and probabilities")
			return no_mass_data
		return mass_data["bin_edges"], mass_data["probabilities"]

	def prob_in_range(self, mass_data, lo: float, hi: float):
		"""Sum the probability in the bins of the chirp mass distribution which lie entirely within
		[lo, hi] (hi may be math.inf). This is conservative: if lo or hi does not coincide with a bin
		edge, the bin straddling it is excluded, so the result underestimates the probability.
		Returns zero if the mass estimate data is not populated.
		"""
		if mass_data is None:
			return 0.
		edges, probabilities = mass_data
		tolerance = 1e-6
		for bound in (lo, hi):
			if edges[0] < bound < edges[-1] and \
			  not any(abs(edge - bound) <= tolerance for edge in edges):
				logger.warning(f"  Chirp mass bound {bound} M☉ is not a bin edge; "
				               "the straddling bin is excluded")
		adder = KahanAdder() # overkill, but why not
		used = []
		for lower_edge, upper_edge, probability in zip(edges[:-1], edges[1:], probabilities):
			if lower_edge >= lo - tolerance and upper_edge <= hi + tolerance:
				adder += probability
				used.append(f"[{lower_edge}, {upper_edge}]")
		logger.info(f"  Probability of {lo} M☉ <= Mchirp <= {hi} M☉: {float(adder)} "
		            f"(bins used: {', '.join(used) if used else 'none'})")
		return float(adder)

	
	def should_follow_up(self, message, metadata):
		alert_type = message["alert_type"].upper()
		if alert_type != self.allowed_alert_type:
			return False, {}
		
		event = message["event"]
		# Some searches (e.g. SSM) send an empty classification and only some of the properties
		classification = event.get("classification") or {}
		properties = event.get("properties") or {}
		p_bns_nsbh = classification.get("BNS", 0.0) + classification.get("NSBH", 0.0)
		far = event["far"]
		t0 = event["time"]
		
		raw_map=astropy.table.Table.read(BytesIO(event["skymap"]))
		skymap = Skymap(raw_map["PROBDENSITY"], raw_map["UNIQ"])
		mean_dist = raw_map.meta.get("DISTMEAN", -1.0)
		prob_area, min_dec = skymap.area_for_probability(GW_CRED)
		prob_area_deg2 = prob_area/DEG2_TO_SR
		mass_data = self.get_chirp_mass_estimate(message, metadata)
		
		logger.info(f"LVK alert with 90% probability area of {prob_area} sr ({prob_area_deg2:.1f} deg²)")
		logger.info(f"    Search: {event.get('search')}, FAR: {far} Hz")
		logger.info(f"    Mean distance: {mean_dist} Mpc")
		logger.info(f"    Minimum declination: {min_dec} radians")
		
		if min_dec > math.radians(MIN_DEC_CUT_DEG): # standard visibility cut
			return False, {}
		
		mass_probs = {}
		def p_mchirp(lo, hi):
			"""Chirp mass range probabilities, computed only when needed and only once"""
			if (lo, hi) not in mass_probs:
				mass_probs[(lo, hi)] = self.prob_in_range(mass_data, lo, hi)
			return mass_probs[(lo, hi)]
		
		result_data = {"skymap": skymap, "90%_area": prob_area, "passed_types": []}
		
		def accept(category, reward_mask, description):
			# Later categories take precedence, overwriting the type and reward map of earlier ones
			result_data["type"] = category
			result_data["reward_mask"] = reward_mask
			result_data["passed_types"].append(category)
			logger.info(f"LVK alert meets criteria for {category} {description}")
		
		is_ssm_search = event.get("search") == SSM_SEARCH
		
		# Binary Neutron Star Mergers and Neutron Star - Black Hole Mergers (§2.1.3)
		# Requirements:
		# - Only trigger on an Initial alert
		# - Step 1, any of:
		#   - P(BNS) + P(NSBH) > 0.9
		#   - HasNS >= 0.5
		#   - HasRemnant >= 0.10
		#   - P(Mchirp < 2.3 M☉) >= 0.9
		#   - P(2.3 M☉ < Mchirp < 5.5 M☉) >= 0.9 and P(BNS) + P(NSBH) > 0.1
		# - Step 2: FAR < 1 per year
		# - Step 3: Omega_obs, the observable part of the 90% credible region, trimmed to its
		#   highest-probability 1500 deg² if larger
		# - Step 4: probability in Omega_obs with |b| > 10 degrees greater than 0.10
		# Further categorization by Omega_obs area (step 5):
		# - Gold: < 100 deg²
		# - Silver: < 500 deg²
		# - Bronze: < 1500 deg²
		# Events with Omega_obs much larger than 1500 deg² are left to the ToO advisory board.
		# Events from the SSM search are considered only under the sub-solar mass criteria below,
		# since their chirp masses would otherwise satisfy step 1 here.
		if is_ssm_search:
			logger.info("    SSM search event: not considered as a BNS or NSBH merger")
		elif far < FAR_1_PER_YR and \
		  (p_bns_nsbh > BNS_NSBH_CLASS_MIN or
		   properties.get("HasNS", 0.0) >= HASNS_MIN or
		   properties.get("HasRemnant", 0.0) >= HASREMNANT_MIN or
		   p_mchirp(0., MCHIRP_NS_MAX) >= MCHIRP_PROB_MIN or
		   (p_mchirp(MCHIRP_NS_MAX, MCHIRP_GAP_MAX) >= MCHIRP_PROB_MIN and
		    p_bns_nsbh > BNS_NSBH_GAP_CLASS_MIN)):
			untrimmed_area = observable_credible_area(skymap, t0, GW_CRED, **self.obs_settings)
			if untrimmed_area > BNS_MAX_AREA_DEG2:
				logger.info(f"    Omega_obs of {untrimmed_area:.1f} deg² is larger than "
				            f"{BNS_MAX_AREA_DEG2} deg²: only the most probable part will be "
				            "targeted, and any further response is left to the ToO advisory board")
			mask, area, obs_prob, high_b_prob = trim_to_observable(skymap, t0, BNS_MAX_AREA_DEG2,
			                                                       GW_CRED, **self.obs_settings)
			logger.info(f"    BNS/NSBH Omega_obs: {area:.1f} deg², probability {obs_prob:.3f}, "
			            f"probability with |b| > {GAL_LAT_CUT_DEG}°: {high_b_prob:.3f}")
			if high_b_prob > HIGH_B_PROB_MIN:
				if area < BNS_GOLD_DEG2:
					category = "GW_case_Gold"
				elif area < BNS_SILVER_DEG2:
					category = "GW_case_Silver"
				else:
					category = "GW_case_Bronze"
				accept(category, mask, "BNS or NSBH merger")
				result_data["bns_nsbh_passed"] = True
				self.event_cache.add(LVK_BNS_NSBH_EVENT, message["superevent_id"], t0,
				                     skymap.credible_mask(GW_CRED, PROB_MAP_ORDER))
		
		# Gravitationally lensed Binary Neutron Star mergers (§5.1.3)
		# Requirements:
		# - "Only trigger on an Initial human-vetted GW detections"
		# - "probability that the GW source includes one or more compact objects in the range
		#   3 – 5 M☉ of no less than 90%: p(HasMassGap)>=0.9", or P(2.3 M☉ < Mchirp < 5 M☉) >= 0.9
		#   (evaluated up to the nearest bin edge, 5.5 M☉)
		# - probability that the GW source is a NS-BH merger of less than 10% : p(NS-BH)<0.1
		# - "False alarm rate less than 1 per 1 year"
		# - "90% credible GW sky localization of no more than 900 degree^2"
		#
		# Further categorization:
		# - Gold: 90% area < 15 square degrees
		# - Silver: 90% area <= 900 square degrees
		if (properties.get("HasMassGap", 0.0) >= LENSED_MASSGAP_MIN or
		    p_mchirp(LENSED_MCHIRP_MIN, LENSED_MCHIRP_MAX) >= MCHIRP_PROB_MIN) and \
		  classification.get("NSBH", 0.0) < LENSED_NSBH_MAX and \
		  far < FAR_1_PER_YR and \
		  prob_area_deg2 <= LENSED_MAX_DEG2:
			mask = trim_to_observable(skymap, t0, LENSED_MAX_DEG2, GW_CRED, **self.obs_settings)[0]
			if not mask.any():
				logger.warning("    No part of the lensed BNS credible region is observable")
			accept("lensed_BNS_case_B" if prob_area_deg2 < LENSED_GOLD_DEG2 else "lensed_BNS_case_A",
			       mask, "lensed BNS merger")

		# Black Hole-Black Hole Mergers (§2.2.3)
		# Requirements:
		# - P(Mchirp >= 22 M☉) >= 0.9
		# - FAR < 1 per year
		# - Full 90% credible area < 100 deg²
		# - Fraction of the total probability lying in observable pixels > 0.70
		if p_mchirp(BBH_MCHIRP_MIN, math.inf) >= MCHIRP_PROB_MIN and \
		  far < FAR_1_PER_YR and \
		  prob_area_deg2 < BBH_MAX_AREA_DEG2:
			prob = skymap.flat_prob_map(PROB_MAP_ORDER)
			obs_fraction = float(prob[observable_mask(1<<PROB_MAP_ORDER, t0, **self.obs_settings)].sum())
			logger.info(f"    BBH observable probability fraction: {obs_fraction:.3f}")
			if obs_fraction > BBH_OBS_FRAC_MIN:
				mask = trim_to_observable(skymap, t0, BBH_MAX_AREA_DEG2, GW_CRED, **self.obs_settings)[0]
				accept("BBH", mask, "binary black hole merger")
		
		# Sub-solar mass candidates (§2.3.3)
		# Requirements:
		# - Event from the SSM search
		# - HasSSM >= 0.5 or P(Mchirp < 0.87 M☉) >= 0.9
		# - FAR < 1 per 3 years
		# - Omega_obs no larger than 500 deg² (larger, Bronze, events are not automated and are left
		#   to the ToO advisory board)
		# - Probability in Omega_obs with |b| > 10 degrees greater than 0.10
		# Further categorization by Omega_obs area:
		# - Gold: < 100 deg²
		# - Silver: < 500 deg²
		if is_ssm_search and \
		  (properties.get("HasSSM", 0.0) >= HASSSM_MIN or
		   p_mchirp(0., MCHIRP_SSM_MAX) >= MCHIRP_PROB_MIN) and \
		  far < FAR_1_PER_3YR:
			untrimmed_area = observable_credible_area(skymap, t0, GW_CRED, **self.obs_settings)
			if untrimmed_area > SSM_MAX_AREA_DEG2:
				logger.info(f"    SSM Omega_obs of {untrimmed_area:.1f} deg² is larger than "
				            f"{SSM_MAX_AREA_DEG2} deg² (Bronze): not triggering automatically, "
				            "left to the ToO advisory board")
			else:
				mask, area, obs_prob, high_b_prob = trim_to_observable(skymap, t0, SSM_MAX_AREA_DEG2,
				                                                       GW_CRED, **self.obs_settings)
				logger.info(f"    SSM Omega_obs: {area:.1f} deg², probability {obs_prob:.3f}, "
				            f"probability with |b| > {GAL_LAT_CUT_DEG}°: {high_b_prob:.3f}")
				if high_b_prob > HIGH_B_PROB_MIN:
					accept("SSM_Gold" if area < SSM_GOLD_DEG2 else "SSM_Silver", mask,
					       "sub-solar mass merger")
		
		passes = len(result_data["passed_types"]) > 0
		if passes:
			logger.info(f"LVK alert passed categories {result_data['passed_types']}; "
			            f"using {result_data['type']}")
		
		# TODO: implement unidentified source alerts
		# Further categorization:
		# - Gold: 90% area < 100 square degrees (0.030461 sr)
		# - Silver: 90% area < 500 square degrees (0.152308 sr)
		if not passes and \
		   prob_area < 0.152308:
			result_data["type"] = "GW_case_C" if prob_area < 4.569261e-3 else "GW_case_E"
			logger.warning("Alert might pass Unidentified Source conditions for type "
			               f"{result_data['type']}, but these are not definitely implemented")
		
		return passes, result_data
	
	def generate_scheduling_data(self, message, metadata, alert_data):
		target_order = 5
		# The reward map is the (trimmed) Omega_obs region of the category which was selected
		flat_map = downgrade_mask(alert_data["reward_mask"], PROB_MAP_ORDER, target_order)
		return {"instrument": message["event"]["instruments"],
		        "alert_type": alert_data["type"],
		        "event_trigger_timestamp": message["event"]["time"],
		        "reward_map": flat_map,
		        "reward_map_nside": 1<<target_order,
		        }

	def after_send(self, message, metadata, alert_data, alert_id, is_test: bool):
		"""
		Neutrino alerts usually arrive long before the LVK Initial alert for the same event, so when
		a BNS/NSBH event passes, look for earlier neutrino alerts which coincide with it, and have
		their filters send neutrino_coincident alerts for them.
		"""
		if not alert_data.get("bns_nsbh_passed", False):
			return
		gw_mask = alert_data["skymap"].credible_mask(GW_CRED, PROB_MAP_ORDER)
		for entry in self.event_cache.find(NEUTRINO_EVENT, message["event"]["time"], NU_COINC_DT_S):
			if numpy.any(entry["mask"] & gw_mask):
				entry["filter"].upgrade_to_coincident(entry, alert_id, is_test)


# This filter should be applicable to other neutrino observatories using the same schema, 
# maybe rename.
class IceCubeAlertFilter(AlertFilter):
	def __init__(self, history: dict, sender: AlertSender, allow_tests: bool=False,
	             alert_type="initial", obs_window_h: float=OBS_WINDOW_H,
	             obs_sun_alt_max_deg: float=OBS_SUN_ALT_MAX_DEG,
	             obs_min_alt_deg: float=OBS_MIN_ALT_DEG, enable_coincidence: bool=False,
	             event_cache: RecentEventCache=None):
		"""
		Args:
		    obs_window_h, obs_sun_alt_max_deg, obs_min_alt_deg: Observability settings used to
		        check that the localisation centroid can be observed.
		    enable_coincidence: Whether to look for coincidences with LVK BNS/NSBH alerts which have
		                        passed filtering, to produce neutrino_coincident alerts. When enabled,
		                        neutrinos which pass the common criteria with p_astro > 0.3 are also
		                        cached, so that an LVK alert which passes later can upgrade them.
		    event_cache: Where to look for and store events for coincidence checks. Defaults to the
		                 shared module-level cache.
		"""
		super().__init__(history, sender, allow_tests)
		self.allowed_alert_type = alert_type
		if alert_type != "initial":
			logger.warning(f"IceCubeAlertFilter alert type is {alert_type}, not initial")
		self.obs_settings = {"window_h": obs_window_h, "sun_alt_max": obs_sun_alt_max_deg,
		                     "min_alt": obs_min_alt_deg}
		self.enable_coincidence = enable_coincidence
		self.event_cache = event_cache if event_cache is not None else recent_events
	
	def is_test(self, message, metadata):
		if super().is_test(message, metadata):
			return True
		return message["alert_tense"] == "test" or message["alert_tense"] == "injection"
	
	def alert_identifier(self, message, metadata):
		# TODO: The GCN schema allows for a list of names, which can make things tricky.
		#       For now, we hope that the list contains only one item.
		return message["event_name"][0], \
		       {"type": message["alert_type"], "time": message["alert_datetime"]}
	
	def overrides_previous(self, old_meta, new_meta):
		# Retractions override previous alerts.
		# If there were an interest in handling updates, etc., logic should be added here.
		return new_meta["type"] == "retraction"
	
	def should_follow_up(self, message, metadata):
		if message["alert_tense"] in ["archival", "planned"]:
			return False, {}
		alert_id, id_meta = self.alert_identifier(message, metadata)
		if message["alert_type"] == "retraction":
			# a retracted neutrino must not be followed up later due to a coincidence
			self.event_cache.remove(NEUTRINO_EVENT, alert_id)
		if message["alert_type"] != self.allowed_alert_type:
			return False, {}
		# "ToOs should not be performed if an alert is retracted"
		if message["alert_type"] == "retraction" or \
		  (self.history.get(alert_id) or {}).get("type") == "retraction":
			return False, {}
		# Maps including systematic uncertainties are expected only with later 'update' alerts, so
		# this is required only when not processing 'initial' alerts.
		if self.allowed_alert_type != "initial" and not message.get("systematic_included", False):
			return False, {}
		
		# Requirements (§3):
		# - Not retracted
		# - Galactic latitude: |b| > 10 degrees
		# - Airmass < 2 at the localisation centroid, while the Sun is below -12 degrees, within
		#   24 hours
		# - >= 90% of the reported contour covered by a single Rubin pointing
		#   This should be seriously evaluated by the scheduler, but we can make a conservative,
		#   approximate cut here
		# Further categorization, tested in this order:
		# - EHE: energy > 1 PeV and p_astro > 0.5
		# - Coincident: p_astro > 0.3, with the 90% contour overlapping that of an LVK BNS/NSBH
		#   alert which passed filtering, and |dT| < 10 minutes
		# - Standard: p_astro > 0.4
		
		p_astro = message.get("p_astro") or 0.0
		min_p_astro = min(NU_STD_PASTRO, NU_EHE_PASTRO)
		if self.enable_coincidence:
			min_p_astro = min(min_p_astro, NU_COINC_PASTRO)
		if p_astro <= min_p_astro:
			return False, {}
		
		pos = astropy.coordinates.ICRS(ra=message["ra"]*astropy.units.deg,
		                               dec=message["dec"]*astropy.units.deg)
		pos_gal = pos.transform_to(astropy.coordinates.Galactic())
		if abs(pos_gal.b.deg) <= GAL_LAT_CUT_DEG:
			return False, {}
		
		t0 = message["trigger_time"]
		if not _observable(message["ra"], message["dec"], t0, **self.obs_settings)[0]:
			logger.info("Neutrino alert centroid is not observable at airmass < 2 within "
			            f"{self.obs_settings['window_h']} hours")
			return False, {}
		
		# get skymap via separate HTTP
		compressed_skymap_data = fetch_file(message["healpix_url"], use_file_cache)
		t = astropy.table.Table.read(BytesIO(gzip.decompress(compressed_skymap_data)))
		ordering = t.meta.get("ORDERING").upper()
		
		if ordering == "NUNIQ":
			skymap = Skymap(t["PROBDENSITY"], t["UNIQ"])
		else:
			# Convert a flat skymap of probabilities to set of non-trivial UNIQ pixels with values
			# of probability/area, dropping pixels which are NaN or less than epsilon along the way.
			map_data = t["PROB"]
			# make sure the map pixel data is a one-dimensional array
			map_data=map_data.reshape((numpy.prod(map_data.shape),))
			nside=int(math.sqrt(len(map_data)/12))
			order = int(math.log2(nside))
			pixel_area = math.pi/(3<<(order<<1))
			if nside*nside*12 != len(map_data):
				raise RuntimeError(f"Invalid number of map pixels: {len(map_data)}")
			pixel_epsilon = 1e-16
			if ordering=="RING":
				base = 4<<(2*order)
				mask = map_data > pixel_epsilon
				useful_pixels = map_data[mask]/pixel_area
				indices = mask.nonzero()[0]
				skymap = Skymap(useful_pixels, base+healpy.ring2nest(nside, indices))
			elif ordering in ("NEST", "NESTED"):
				base = len(map_data)//3
				mask = map_data > pixel_epsilon
				useful_pixels = map_data[mask]/pixel_area
				indices = mask.nonzero()[0]
				skymap = Skymap(useful_pixels, base+indices)
			else:
				raise RuntimeError(f"Unexpected healpix ordering: {ordering}")
		
		prob_area, min_dec = skymap.area_for_probability(NU_CRED)
		logger.info(f"Neutrino alert with 90% probability area of {prob_area} sr, "
		            f"minimum declination: {min_dec} radians")
		
		if min_dec > math.radians(MIN_DEC_CUT_DEG): # standard visibility cut
			return False, {}
		
		# If the area for 90% of the probability is greater than the camera field of view,
		# there is no single exposure which can capture it.
		# This does not, however, rule out cases in which the area is smaller than the field of
		# view, but distributed in such a way that it cannot be fit inside the shape of the field
		# of view.
		if prob_area > RUBIN_FOV_SR:
			return False, {}
		
		if self.enable_coincidence:
			# Remember this neutrino, whichever category it falls into (if any), so that if an LVK
			# BNS/NSBH alert for a coincident event arrives later it can be upgraded to
			# neutrino_coincident.
			previous = self.event_cache.get(NEUTRINO_EVENT, alert_id)
			self.event_cache.add(NEUTRINO_EVENT, alert_id, t0, skymap.credible_mask(NU_CRED, PROB_MAP_ORDER),
			                     filter=self, message=message, metadata=metadata,
			                     alert_data={"skymap": skymap, "90%_area": prob_area},
			                     id_meta=id_meta, is_test=self.is_test(message, metadata),
			                     sent_type=previous["sent_type"] if previous is not None else None)
		
		energy = message.get("nu_energy") or 0.0  # TeV
		category = None
		if energy > NU_EHE_ENERGY_TEV and p_astro > NU_EHE_PASTRO:
			category = "neutrino_EHE"
		elif self.enable_coincidence and p_astro > NU_COINC_PASTRO and \
		  self.find_coincident_gw(skymap, t0):
			category = "neutrino_coincident"
		elif p_astro > NU_STD_PASTRO:
			category = "neutrino"
		if category is None:
			return False, {}
		
		result_data = {"skymap": skymap, "90%_area": prob_area, "type": category}
		logger.info(f"Neutrino alert meets criteria for {result_data['type']}")
		
		return True, result_data
	
	def find_coincident_gw(self, skymap: Skymap, t0):
		"""
		Look for LVK BNS/NSBH events which passed filtering within NU_COINC_DT_S of t0 and whose 90%
		credible regions overlap the neutrino's.
		
		This can only find LVK events which have already been processed. LVK Initial alerts usually
		arrive long after neutrino alerts, so the reverse case is handled by LVKAlertFilter.after_send
		and upgrade_to_coincident.
		"""
		nu_mask = skymap.credible_mask(NU_CRED, PROB_MAP_ORDER)
		for entry in self.event_cache.find(LVK_BNS_NSBH_EVENT, t0, NU_COINC_DT_S):
			if numpy.any(nu_mask & entry["mask"]):
				logger.info(f"Neutrino alert is coincident with {entry['source']}")
				return True
		return False
	
	def after_send(self, message, metadata, alert_data, alert_id, is_test: bool):
		# record what has been sent for this neutrino, in case it is later upgraded
		entry = self.event_cache.get(NEUTRINO_EVENT, alert_id)
		if entry is not None:
			entry["sent_type"] = alert_data["type"]
	
	def upgrade_to_coincident(self, entry: dict, gw_source: str, gw_is_test: bool):
		"""
		Send a neutrino_coincident alert for a neutrino from the recent event cache which has been
		found to coincide with an LVK BNS/NSBH event which passed filtering after the neutrino alert
		was processed. If a neutrino alert was already sent for it, the new alert is marked as an
		update; EHE neutrinos and neutrinos already marked as coincident are left as they are.
		
		Return: Whether an alert was sent
		"""
		if entry["sent_type"] in ("neutrino_EHE", "neutrino_coincident"):
			return False
		alert_data = dict(entry["alert_data"], type="neutrino_coincident")
		is_update = entry["sent_type"] is not None
		logger.info(f"Neutrino {entry['source']} is coincident with {gw_source}; sending "
		            f"{alert_data['type']} {'update' if is_update else 'alert'}")
		self.history[entry["source"]] = entry["id_meta"]
		scheduling_data = self.generate_scheduling_data(entry["message"], entry["metadata"], alert_data)
		self.send_scheduling_data(scheduling_data, entry["source"], entry["is_test"] or gw_is_test,
		                          is_update)
		entry["sent_type"] = alert_data["type"]
		return True
	
	def generate_scheduling_data(self, message, metadata, alert_data):
		target_order = 5
		flat_map = alert_data["skymap"].make_flat_binary_map(NU_CRED, target_order)
		return {"instrument": [message["mission"]],
		        "alert_type": alert_data["type"],
		        "event_trigger_timestamp": message["trigger_time"],
		        "reward_map": flat_map,
		        "reward_map_nside": 1<<target_order,
		        }


# this might or might not be replaced by a more general supernova filter
class SuperKAlertFilter(AlertFilter):
	def __init__(self, history: dict, sender: AlertSender, allow_tests: bool=False,
	             alert_type="initial"):
		super().__init__(history, sender, allow_tests)
		self.allowed_alert_type = alert_type
	
	def is_test(self, message, metadata):
		if super().is_test(message, metadata):
			return True
		return message["alert_tense"] == "test" or message["alert_tense"] == "injection"
	
	def alert_identifier(self, message, metadata):
		# SuperK alerts do not appear to have event names, but do seem to have an id
		# ids can be either a string or a list of strings
		if isinstance(message["id"], str):
			id = message["id"]
		else:
			id = message["id"][0]
		return id, {"type": message["alert_type"], "time": message["alert_datetime"]}
	
	def overrides_previous(self, old_meta, new_meta):
		# Retractions override previous alerts.
		# If there were an interest in handling updates, etc., logic should be added here.
		return new_meta["type"] == "retraction"
	
	def should_follow_up(self, message, metadata):
		if message["alert_tense"] in ["archival", "planned"]:
			return False, {}
		# Criteria:
		# - Search Area maximum 100 sq deg
		
		# For now, the only data we have a is a circular 68% region, so we use that
		containment_radius = message["ra_dec_error"]
		prob_area = math.pi*containment_radius*containment_radius
		
		if prob_area > 100.:
			return False, {}
		
		return True, {}

	def generate_scheduling_data(self, message, metadata, alert_data):
		target_order = 5
		flat_n_npixels=12<<(target_order<<1)
		flat_map = numpy.zeros(flat_n_npixels, dtype=bool)
		
		# find the indices of all pixels touched by the localization circle
		@dataclass
		class Deg:
			deg: float
			def to_value(self, *ignored):
				return self.deg
		indices = healpix_cone_search(lon=Deg(message["ra"]), lat=Deg(message["dec"]),
		                              radius=Deg(message["ra_dec_error"]),
		                              nside=1<<target_order, order="nested")
		# mark each touched pixel
		for idx in indices:
			flat_map[idx] = True
		
		return {"instrument": [message["mission"]],
		        "alert_type": "SN_Galactic",
		        "event_trigger_timestamp": message["trigger_time"],
		        "reward_map": flat_map,
		        "reward_map_nside": 1<<target_order,
		        }


input_constructors = {
	"files": FileConsumer,
	"kafka": KafkaConsumer,
}

# TODO: The 2026 strategy also defines GRB cases, lensed GRBs, and solar system objects, which have
#       no filters yet.
filter_constructors = {
	"lvk_gw": LVKAlertFilter,
	"icecube_nu": IceCubeAlertFilter,
	"superk_sn": SuperKAlertFilter,
}

output_constructors = {
	"stdout": StdoutSender,
	"files": FileSender,
	"kafka": KafkaSender,
	"confluent_rest": ConfluentRESTSender,
}

def get_message_contents(message):
	# Blobs, JSON, and Avro messages contain their payloads in a member named 'content'
	if hasattr(message, "content"):
		# Avro messages have a 'single_record' member indicating whether their content is logically
		# a single item (which might be a list), or a list of distinct records.
		# Other message types are inherently single records, and have no explicit attribute.
		if getattr(message, "single_record", True):
			return [message.content]
		else:
			return message.content
	else:
		return [message]

if __name__ == "__main__":
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
						help="mapping of topic names to filter types; supported filter names are: "
						f"{list(filter_constructors.keys())}")
	parser.add_argument("--filter-settings", type=json.loads, default={}, 
	                    help="mapping of filter types to dinstinct settings for each")
	parser.add_argument("--output-type", type=str, choices=output_constructors.keys(), default="stdout",
						help="the mechanism to use for sending passing alert data")
	parser.add_argument("--output-options", type=json.loads, default={}, 
						help="settings for the output sender")
	parser.add_argument("--use-file-cache", action="store_true", default=False, 
						help="Write files fetched via HTTP to a local cache, and read them from the cache if available")
	parser.add_argument("--treat-cache-miss-as-not-found", action="store_true", default=False,
	                    help="When using the local file cache, treat the lack of a cache entry as "
	                    "the file being inaccessible, making no HTTP request for it")
	parser.add_argument("input_files", nargs='*', help="files to be read with the file consumer")

	config = parser.parse_args()

	if config.input_type not in input_constructors:
		logger.fatal(f"Unrecognized input type: {config.input_type}")
		exit(1)
	if config.output_type not in output_constructors:
		logger.fatal(f"Unrecognized output type: {config.output_type}")
		exit(1)
	
	use_file_cache = config.use_file_cache
	treat_cache_miss_as_not_found = config.treat_cache_miss_as_not_found

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
	for filter in filter_constructors.keys():
		settings = {"allow_tests": config.allow_tests}
		if filter in config.filter_settings:
			settings.update(config.filter_settings[filter])
		config.filter_settings[filter] = settings
	filters = {}
	for topic, filter in config.filters.items():
		filters[topic] = filter_constructors[filter](history, sender, **config.filter_settings[filter])

	for message, metadata in consumer.read(metadata=True, autocommit=False):
		if metadata.topic not in filters:
			logger.error(f"Message metadata claims it is from unexpected topic '{metadata.topic}'")
			continue
		for record in get_message_contents(message):
			try:
				filters[metadata.topic].process(record, metadata)
			except Exception as e:
				logger.error(f"Error processing alert: {repr(e)}\nDropping and continuing with next")
		consumer.mark_done(metadata)
