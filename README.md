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
	  icecube_nu:
	    alert_type: "update"

The currently supported per-filter setings are:

#### lvk_gw
- alert_type: The alert type value to process. The default (and recommended) value is "INITIAL".
              Valid values are: "EARLYWARNING", "PRELIMINARY" ," INITIAL" , "UPDATE" , and "RETRACTION"

#### icecube_nu:
- alert_type: The alert type value to process. The default (and recommended) value is "update".
              Valid values are: "initial", "subsequent", "update", and "retraction"

### Event Types

The following set of labels is used in the output records to identify the various cases outlined in the recommednation paper:

	| too_types_to_follow in scheduler 	| Corresponding strategy from ToO 2024 recommendation 	|
	|----------------------------------	|-----------------------------------------------------	|
	| GW_case_A                        	| -                                                   	|
	| GW_case_B                        	| GW gold                                             	|
	| GW_case_C                        	| unidentified gold                                   	|
	| GW_case_D                        	| GW silver                                           	|
	| GW_case_E                        	| Unidentified silver                                 	|
	| BBH_case_A                       	| BBH_dark_near                                       	|
	| BBH_case_B                       	| BBH_dark_far                                        	|
	| BBH_case_C                       	| BBH_bright                                          	|
	| lensed_BNS_case_A                	| 900 deg skymap                                      	|
	| lensed_BNS_case_B                	| 15 deg skymap                                       	|
	| neutrino and neutrino_u          	| Neutrino                                            	|
	| SSO_night                        	| Small PHA                                           	|
	| SSO_twilight                     	| Small PHA                                           	|
	| GW_case_large                    	| Large GW skymaps                                    	|
	| Lensed_GRB                       	| Lensed GRB                                          	|
	| SN_Galactic                      	| Galactic supernova                                  	|
