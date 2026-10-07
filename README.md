Rubin ToO Alert Producer
========================

This is an implementation of the Target of Opportunity Alert Producer (ToO Alert Producer), as described by [TSTN-035: Handling Targets of Opportunity](https://tstn-035.lsst.io/). 
It is intended to listen for alerts published via an external system (e.g. the HOPSKOTCH pub-sub service), apply filtering rules as outlined in [Rubin ToO 2024:
Envisioning the Vera C. Rubin Observatory LSST Target of Opportunity program](https://arxiv.org/abs/2411.04793), and forward suitable descriptions of any passing alerts to the telescope scheduler via the EFD.

Not all alert types are yet implemented. 

### Interface

This tool is inded to work as a long-running service, typically inside a Kubernetes cluster. 
It can therefore be configured from a YAML configuration file which can be injected into its container via a Kubernetes ConfigMap. 
For testing and debugging it can also be run as a stand-alone script, using either a configuration file or command-line options. 
Input and output are handled using pluggable mechanisms, so that input can be from some set of topics on an Apache Kafka cluster, or a simple set of files, and output can be published to a Kafka topic (which may be on another cluster than the input topics), routed indirectly to a Kafka cluster via an instance of the Confluent Kafka REST Proxy, or written (in a reduced form) to standrd output. 
Each input topic should be mapped in the configuration to an alert filter type. 

An example configuration file might look like:

	# Ignore alerts marked by the sender(s) as tests
	allow-tests: false
	
	# Listen for alerts from two Kafka topics, topic1 and topic2
	input-type: "kafka"
	input-options:
	  url: "kafka://127.0.0.1:9092/topic1,topic2"
	
	# Treat the alerts from both topics as gravitational wave alerts in the LVK format
	filters:
	  topic1: lvk_gw
	  topic2: lvk_gw
	
	# Write passing alerts to topic3 of some Kafka broker via a REST proxy
	output-type: "confluent_rest"
	output-options:
	  url: "http://localhost:8082/topics/topic3"

Per-filter-type settings can also be specified via the `filter-settings` mapping, e.g.:

	filter-settings:
	  lvk_gw:
	    alert_type: "PRELIMINARY"
	    obs_min_alt_deg: 35
	  icecube_nu:
	    alert_type: "initial"
	    enable_coincidence: true

The currently supported per-filter setings are:

#### lvk_gw
- alert_type: The alert type value to process. The default (and recommended) value is "INITIAL".
              Valid values are: "EARLYWARNING", "PRELIMINARY" ," INITIAL" , "UPDATE" , and "RETRACTION"
- obs_window_h: The length, in hours from the event time, of the window in which a sky position must
                be observable to count towards Omega_obs. The default is 24.
- obs_sun_alt_max_deg: The maximum altitude of the Sun, in degrees, for a time to count as night.
                       The default is -12.
- obs_min_alt_deg: The minimum altitude, in degrees, at which a sky position counts as observable.
                   The default is 30 (airmass 2).

#### icecube_nu:
- alert_type: The alert type value to process. The default (and recommended) value is "initial".
              Valid values are: "initial", "subsequent", "update", and "retraction".
              Alerts which do not include systematic uncertainties are only accepted when this is
              "initial".
- obs_window_h, obs_sun_alt_max_deg, obs_min_alt_deg: Observability settings, as for lvk_gw, applied
              to the localisation centroid.
- enable_coincidence: Whether to look for coincidences with LVK BNS/NSBH alerts, producing
                      neutrino_coincident alerts. The default is false.
                      A neutrino arriving after a coincident LVK alert has passed is sent as
                      neutrino_coincident directly. Since LVK Initial alerts usually arrive long after
                      neutrino alerts, neutrinos passing the common criteria with p_astro > 0.3 are
                      also remembered for 6 hours (by event time), and if a coincident LVK BNS/NSBH
                      alert passes later a neutrino_coincident alert is sent for the neutrino: marked
                      as an update if a neutrino alert was already sent for it, or as a new alert if
                      it was too weak to be sent on its own. EHE neutrinos are not changed.

### Event Types

The following set of labels is used in the output records to identify the various cases outlined in the recommendation papers. 
Section numbers refer to the Rubin ToO 2026 revised strategy report.
Labels marked as retired are no longer produced, but are kept here for reference.

	| too_types_to_follow in scheduler 	| Corresponding strategy                                            	|
	|----------------------------------	|-------------------------------------------------------------------	|
	| GW_case_Gold                     	| 2026 BNS/NSBH gold: Omega_obs < 100 deg² (§2.1.3)                 	|
	| GW_case_Silver                   	| 2026 BNS/NSBH silver: Omega_obs < 500 deg² (§2.1.3)               	|
	| GW_case_Bronze                   	| 2026 BNS/NSBH bronze: Omega_obs < 1500 deg², after trimming (§2.1.3)	|
	| BBH                              	| 2026 BBH: 90% area < 100 deg² (§2.2.3)                            	|
	| SSM_Gold                         	| 2026 sub-solar mass gold: Omega_obs < 100 deg² (§2.3.3)           	|
	| SSM_Silver                       	| 2026 sub-solar mass silver: Omega_obs < 500 deg², after trimming (§2.3.3)	|
	| lensed_BNS_case_A                	| Lensed BNS, 900 deg² skymap (§5.1.3)                              	|
	| lensed_BNS_case_B                	| Lensed BNS, 15 deg² skymap (§5.1.3)                               	|
	| neutrino                         	| 2026 standard neutrino: p_astro > 0.4 (§3)                        	|
	| neutrino_EHE                     	| 2026 extremely high energy neutrino: > 1 PeV, p_astro > 0.5 (§3)  	|
	| neutrino_coincident              	| 2026 neutrino coincident with an LVK BNS/NSBH alert (§3); may be an update to a neutrino alert	|
	| neutrino_u                       	| Neutrino                                                          	|
	| SN_Galactic                      	| Galactic supernova                                                	|
	| GW_case_A                        	| -                                                                 	|
	| GW_case_C                        	| 2024 unidentified gold (not yet implemented)                      	|
	| GW_case_E                        	| 2024 unidentified silver (not yet implemented)                    	|
	| BBH_case_B                       	| 2024 BBH_dark_far (not implemented)                               	|
	| BBH_case_C                       	| 2024 BBH_bright (not implemented)                                 	|
	| SSO_night                        	| Small PHA (not implemented)                                       	|
	| SSO_twilight                     	| Small PHA (not implemented)                                       	|
	| Lensed_GRB                       	| Lensed GRB (not implemented)                                      	|
	| GW_case_B                        	| Retired: 2024 GW gold, replaced by GW_case_Gold                   	|
	| GW_case_D                        	| Retired: 2024 GW silver, replaced by GW_case_Silver               	|
	| BBH_case_A                       	| Retired: 2024 BBH_dark_near, replaced by BBH                      	|
	| GW_large                         	| Retired: 2024 large GW skymaps; Omega_obs > 1500 deg² is trimmed to GW_case_Bronze, and any further response is left to the ToO advisory board	|

Omega_obs is the part of the 90% credible region which is observable from Rubin (Sun below -12°, airmass < 2) within 24 hours of the event. 
When it is larger than a category's maximum area, only its highest-probability pixels within that area are kept. 
Probabilities are not renormalised to the observable region. 
For all GW categories the reward map is this (trimmed) Omega_obs region, as a binary map. 
Sub-solar mass events require HasSSM >= 0.5 from the SSM search; Omega_obs larger than 500 deg² is trimmed to 500 deg² and issued as SSM_Silver.
