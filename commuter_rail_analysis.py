import pandas as pd
import googlemaps
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.stats import linregress
import statsmodels.api as sm
import warnings
import sys
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# number of samples from Gaussian distribution 
samples = 10000

# draw random sample from normal distribution
def univariate_Gaussian_sample(mean, sd, samples):
	return(np.array(list(map(lambda x : max(x, 0), np.random.normal(loc=mean, scale=sd, size=samples)))))

# draw random sample from multivariate normal distribution
def multivariate_Gaussian_sample(mean, sd):
	return(np.array(list(map(lambda x : max(x, 0), np.random.multivariate_normal(mean, np.diag(sd))))))

# census recognized municipalities on Long Island, Connecticut and NY counties of Westchester, Putnam and Dutchess that are served by Metro North and LIRR
ny_city_names = ['Wilton town', 'White Plains city', 'Westport town', 'West Haven town', 'Waterbury town', 'Amenia town', 'Stratford town', 'Stamford town', 'Southeast town', 'Norwalk town', 'Scarsdale town', 'Rye city', 'Redding town', 'North Salem town', 'Poughkeepsie town', 'Mount Pleasant town', 'Pelham town', 'Peekskill city', 'Pawling town', 'Patterson town', 'Ossining town', 'New Rochelle city', 'New Haven town', 'New Canaan town', 'Mount Kisco town', 'Milford town', 'Mamaroneck town', 'Greenburgh town', 'Harrison town', 'Greenwich town', 'Lewisboro town', 'Fairfield town', 'Dover town', 'Darien town', 'Danbury town', 'Cortlandt town', 'Philipstown town', 'New Castle town', 'Bridgeport town', 'Ridgefield town', 'Bethel town', 'Bedford town', 'Beacon city', 'Babylon town', 'East Hampton town', 'Huntington town', 'Islip town', 'Riverhead town', 'Smithtown town', 'Southampton town', 'Southold town', 'Glen Cove city', 'Long Beach city', 'Oyster Bay town']

# census recognized municipalities in Eastern Mass and Rhode Island that are served by MBTA commuter rail system
boston_city_names = ['Rockport town','Gloucester city','Manchester-by-the-Sea town','Beverly city','Newburyport city','Rowley town','Ipswich town','Hamilton town','Wenham town','Salem city','Swampscott town','Lynn city','Fitchburg city','Leominster city','Shirley town','Ayer town','Littleton town','Acton town','Concord town','Lincoln town','Weston town','Waltham city','Haverhill city','Lawrence city','Andover town','Reading town','Wakefield town','Lowell city','Billerica town','Wilmington town','Woburn city','Worcester city','Grafton town','Westborough town','Southborough town','Ashland town','Framingham town','Natick town','Wellesley town','Needham town','Franklin Town city','Norfolk town','Walpole town','Norwood town','Westwood town','Kingston town','Halifax town','Hanson town','Whitman town','Abington town','Scituate town','Cohasset town','Hingham town','North Kingstown town','Warwick city','Providence city','Attleboro city','Mansfield town','Sharon town','Stoughton town','Canton town','Middleborough town','Lakeville town','Bridgewater town','Brockton city','Holbrook town','Randolph town']

# extract commuter rail usage from census dataset
percent_pt_ny = []
percent_pt_boston = []
percent_pt_ny_MOE_lower = []
percent_pt_boston_MOE_lower = []
percent_pt_ny_MOE_upper = []
percent_pt_boston_MOE_upper = []
ny_con_int = []
boston_con_int = []
ny_pt_est = [0] * len(ny_city_names)
boston_pt_est = [0] * len(boston_city_names)
ny_pt_se = [0] * len(ny_city_names)
boston_pt_se = [0] * len(boston_city_names)
ny_alone_est = [0] * len(ny_city_names)
boston_alone_est = [0] * len(boston_city_names)
ny_alone_se = [0] * len(ny_city_names)
boston_alone_se = [0] * len(boston_city_names)
ny_carpool_est = [0] * len(ny_city_names)
boston_carpool_est = [0] * len(boston_city_names)
ny_carpool_se = [0] * len(ny_city_names)
boston_carpool_se = [0] * len(boston_city_names)
ny_other_est = [0] * len(ny_city_names)
boston_other_est = [0] * len(boston_city_names)
ny_other_se = [0] * len(ny_city_names)
boston_other_se = [0] * len(boston_city_names)

not_pt_options = list(['Car, truck, or van: Drove alone', 'Car, truck, or van: Carpooled', 'Other travel mode'])

boston_ny_dict = {'percent_pt_list' : [percent_pt_ny, percent_pt_boston],
				  'percent_pt_MOE_list_lower' : [percent_pt_ny_MOE_lower, percent_pt_boston_MOE_lower],
				  'percent_pt_MOE_list_upper' : [percent_pt_ny_MOE_upper, percent_pt_boston_MOE_upper],
				  'city_names' : [ny_city_names, boston_city_names],
				  'database' : ['ctny.csv', 'mari.csv'],
				  'dest' : ['Manhattan borough', 'Boston city'],
				  'confidence interval' : [ny_con_int, boston_con_int],
				  'pt est' : [ny_pt_est, boston_pt_est],
				  'pt se' : [ny_pt_se, boston_pt_se],
				  'alone est' : [ny_alone_est, boston_alone_est],
				  'alone se' : [ny_alone_se, boston_alone_se],
				  'carpool est' : [ny_carpool_est, boston_carpool_est],
				  'carpool se' : [ny_carpool_se, boston_carpool_se],
				  'other est' : [ny_other_est, boston_other_est],
				  'other se' : [ny_other_se, boston_other_se]}

# eztract relevant info from census dataset
for i in range(2):

	df = pd.read_csv(boston_ny_dict['database'][i])

	for j in range(len(boston_ny_dict['city_names'][i])):

		# extract relevant rows
		city_data = df[(df['Municipality'] == boston_ny_dict['city_names'][i][j]) & (df['Municipality_dest'] == boston_ny_dict['dest'][i])]
		not_pt_vector = np.zeros(samples)
		not_pt_num = 0

		# iterate across modes of transportation
		for _,row in city_data.iterrows():
			est = float(row['Number'].replace(",",""))  

			# convert margin of error associated with 90% confidence interval to standard error
			se = float(row['MOE'].replace(",","")) / 1.645
			
			if row['Mode'] == 'Car, truck, or van: Drove alone':
				not_pt_vector += univariate_Gaussian_sample(est, se, samples)
				not_pt_num += est

				boston_ny_dict['alone est'][i][j] = est
				boston_ny_dict['alone se'][i][j] = round(se,4)

			elif row['Mode'] == 'Car, truck, or van: Carpooled':
				not_pt_vector += univariate_Gaussian_sample(est, se, samples)
				not_pt_num += est

				boston_ny_dict['carpool est'][i][j] = est
				boston_ny_dict['carpool se'][i][j] = round(se,4)

			elif row['Mode'] == 'Other travel mode':
				not_pt_vector += univariate_Gaussian_sample(est, se, samples)
				not_pt_num += est

				boston_ny_dict['other est'][i][j] = est
				boston_ny_dict['other se'][i][j] = round(se,4)

			else:
				pt_vector = univariate_Gaussian_sample(est, se, samples)
				pt_num = est

				boston_ny_dict['pt est'][i][j] = est
				boston_ny_dict['pt se'][i][j] = round(se,4)

		# point estimate for proportion of commuters that use the commuter rail
		pt_percent_num = pt_num / (pt_num + not_pt_num)

		# random samples of the proportion of commuters that use the commuter rail
		pt_percent_vector = np.nan_to_num(pt_vector / (pt_vector + not_pt_vector))
		
		# 0.05 quantile of random samples corresponding to lower bound of 90% confidence interval
		pt_percent_MOE_lower = pt_percent_num - max(0.0, np.quantile(pt_percent_vector, 0.05))

		# 0.95 quantile of random samples corresponding to upper bound of 90% confidence interval
		pt_percent_MOE_upper = min(1.0, np.quantile(pt_percent_vector, 0.95)) - pt_percent_num

		boston_ny_dict['percent_pt_list'][i].append(round(pt_percent_num,4))
		boston_ny_dict['percent_pt_MOE_list_lower'][i].append(round(pt_percent_MOE_lower,4))
		boston_ny_dict['percent_pt_MOE_list_upper'][i].append(round(pt_percent_MOE_upper,4))
		boston_ny_dict['confidence interval'][i].append(round((pt_percent_MOE_upper + pt_percent_MOE_lower), 4))

# distance in miles from station to terminal station (Grand Central Terminal, Penn Station, South Station, North Station)

# Southeast
SE = 53.2
# North White Plains
NWP = 24
wassaic = 81.4
poughkeepsie = 73.5
croton_harmon = 33.2
stamford = 33.0
new_haven = 72.3
danbury = 64.9
wachusett = 53.7
newportbury = 36.2
rockport = 35.3
lowell = 25.5
haverhill = 32.9
worcester = 44.2 
framington = 21.4
providence = 43.6
stoughton = 18.9
kingston = 35.1 
middleborough = 35.6
forge_park = 30.3
greenbush = 27.6
huntington = 34.7
port_washington = 18.1
babylon = 36.6
far_rockaway = 20.4
long_beach = 22.7
ronkonkoma = 48.5

# time, distance and number of stops of Metro North, LIRR and MBTA lines
harlem_line_electric = {'time' : [51, 86, 52, 84, 51, 40, 83, 52, 40, 84, 43, 87, 81, 58, 45, 92, 83, 62, 43, 87, 42, 89, 58, 84, 42, 83, 46, 97], 
			   			'distance' : [NWP, SE, NWP, SE, NWP, NWP, SE, NWP, NWP, SE, NWP, SE, SE, NWP, NWP, SE, SE, NWP, NWP, SE, NWP, SE, NWP, SE, NWP, SE, NWP, SE],
					    'stops' : [14, 14, 16, 14, 14, 5, 13, 16, 4, 14, 4, 14, 8, 16, 5, 14, 7, 16, 2, 10, 1, 10, 16, 10, 4, 11, 5, 14]}
harlem_line_diesel = {'time' : [123, 128, 123, 127], 
					  'distance' : [wassaic] * 4,
					  'stops' : [12, 12, 10, 13]}

hudson_line_electric = {'time' : [63, 57, 65, 57, 65, 50, 69, 58, 52, 49, 71, 51, 70, 52, 60, 52, 67, 52],
						'distance' : [croton_harmon] * 18,
						'stops' : [16, 11, 19, 11, 19, 5, 19, 9, 5, 2, 19, 1, 19, 4, 9, 5, 19, 6]}

hudson_line_diesel = {'time' : [100, 100, 100, 103, 104, 93, 106, 96, 105, 97, 104, 107, 101],
						'distance' : [poughkeepsie] * 13,
						'stops' : [8, 8, 8, 8, 8, 3, 8, 3, 8, 3, 8, 9, 8]}

new_haven_line_electric = {'time' : [110, 117, 112, 109, 118, 111, 107, 126, 107, 125, 106, 126, 115, 108, 67, 67, 68, 69, 59, 70, 60, 61, 73, 61, 60, 73, 60],
						   'distance' : [new_haven] * 14 + [stamford] * 13,
						   'stops' : [12, 16, 12, 9, 13, 8, 5, 17, 6, 17, 6, 18, 15, 8, 14, 14, 14, 14, 7, 14, 7, 7, 14, 6, 6, 14, 8]}

new_haven_line_diesel = {'time' : [125, 117, 119, 125],
						 'distance' : [danbury] * 4,
						 'stops' : [9, 8, 9, 9]}

metro_north_electric = {'time' : harlem_line_electric['time'] + hudson_line_electric['time'] + new_haven_line_electric['time'],
						'distance' : harlem_line_electric['distance'] + hudson_line_electric['distance'] + new_haven_line_electric['distance'],
						'stops' : harlem_line_electric['stops'] + hudson_line_electric['stops'] + new_haven_line_electric['stops']}

metro_north_diesel = {'time' : harlem_line_diesel['time'] + hudson_line_diesel['time'] + new_haven_line_diesel['time'],
						'distance' : harlem_line_diesel['distance'] + hudson_line_diesel['distance'] + new_haven_line_diesel['distance'],
						'stops' : harlem_line_diesel['stops'] + hudson_line_diesel['stops'] + new_haven_line_diesel['stops']}


LIRR_electric = {'time' : [63, 68, 55, 62, 58, 58, 64, 62, 45, 46, 47, 39, 41, 37, 51, 35, 44, 36, 53, 61, 57, 53, 55, 55, 54, 51, 51, 59, 53, 54, 55, 62, 63, 63, 53, 62, 65, 67, 70, 60, 70, 71, 68, 75, 62, 73, 81, 71, 76, 79, 80, 76, 80, 85],
				 'distance' : [huntington] * 8  + [port_washington] * 10 + [far_rockaway] * 6 + [long_beach] * 7 + [babylon] * 15 + [ronkonkoma] * 8,
				 'stops' : [7, 11, 3, 7, 2, 2, 9, 5, 11, 11, 11, 4, 4, 3, 7, 3, 5, 4, 8, 12, 10, 7, 10, 11, 7, 7, 6, 6, 6, 8, 7, 7, 8, 8, 1, 7, 3, 5, 7, 1, 7, 9, 11, 16, 6, 13, 10, 5, 9, 8, 8, 7, 10, 11]}

MBTA_lines = {'time' : [97, 98, 80, 96, 82, 95, 76, 77, 70, 72, 76, 65, 59, 66, 68, 66, 60, 47, 46, 47, 47, 46, 41, 46, 47, 70, 70, 70, 58, 95, 80, 80, 83, 83, 91, 66, 54, 55, 55, 55, 74, 75, 65, 71, 61, 66, 65, 74, 38, 38, 40, 36, 34, 57, 58, 59, 59, 57, 57, 58, 58, 58, 58, 60, 68, 70, 62, 64, 58, 59, 60, 59, 59],
			    'distance' : [wachusett] * 6 + [rockport] * 5 + [newportbury] * 6 + [lowell] * 8 + [haverhill] * 4 + [worcester] * 7 + [framington] * 4 + [providence] * 8 + [stoughton] * 5 + [kingston] * 5 + [middleborough] * 5 + [forge_park] * 5 + [greenbush] * 5,
			    'stops' : [16, 17, 7, 15, 8, 15, 11, 12, 8, 9, 10, 10, 6, 10, 10, 10, 6, 6, 6, 6, 6, 6, 3, 6, 6, 12, 12, 12, 4, 16, 9, 8, 8, 8, 13, 2, 11, 11, 11, 11, 9, 9, 6, 7, 4, 7, 6, 9, 5, 5, 5, 5, 4, 6, 6, 7, 7, 7, 7, 6, 7, 7, 7, 10, 12, 13, 8, 1, 8, 7, 7, 7, 7]}

# average additional time per station for a given commuter rail system
def time_cost_per_station(dic):
	time = dic['time']
	distance = dic['distance']
	stops = dic['stops']
	speed_list = [None] * len(time)
	x_neg = 0
	x_pos = 4

	def compute_slope(x):
		for i in range(len(speed_list)):
			speed_list[i] = distance[i] / ((time[i] - (x * stops[i])) / 60)
		speed_array = np.array(speed_list)
		stops_array = np.array(stops)
		slope,_,_,_,_ = linregress(stops_array, speed_array)
		return(slope, x)

	while True:
		slope, x_val = compute_slope((x_neg + x_pos) / 2)
		if abs(slope) < 0.0001:
			return(x_val)
		if slope > 0:
			x_pos = x_val
		else:
			x_neg = x_val

# average percent time savings due to MBTA electrification
average_electric_station_cost = (time_cost_per_station(metro_north_electric) + time_cost_per_station(LIRR_electric)) / 2
MBTA_time_savings_per_station = time_cost_per_station(MBTA_lines) - average_electric_station_cost
average_MBTA_stations = np.mean(np.array(MBTA_lines['stops']))
average_time_savings = MBTA_time_savings_per_station * average_MBTA_stations
average_time = np.mean(np.array(MBTA_lines['time']))
average_MBTA_percent_time_savings = (average_time_savings / average_time) * 100

# Army Corps of Engineers estimation of time difference between electric and diesel trains on proposed South Coast Rail
south_coast_rail_diesel = [84, 82, 85, 83, 96, 94]
south_coast_rail_electric = [75, 72, 76, 73, 87, 85]

# South Coast Rail predicted time savings from electrification 
south_coast_rail_difference = [None] * 6
for i in range(6):
	south_coast_rail_difference[i] = (south_coast_rail_diesel[i] - south_coast_rail_electric[i]) / south_coast_rail_diesel[i]
south_coast_rail_average_time_savings = 100 * np.mean(np.array(south_coast_rail_difference))

# average number of miles per stop for given commuter rail system
metro_north_average_miles_per_stop = sum(metro_north_electric['distance']) / sum(metro_north_electric['stops'])
LIRR_average_miles_per_stop = sum(LIRR_electric['distance']) / sum(LIRR_electric['stops'])
MBTA_average_miles_per_stop = sum(MBTA_lines['distance']) / sum(MBTA_lines['stops'])

# average speed of given commuter rail system
metro_north_average_speed = 60 * sum(metro_north_electric['distance']) / sum(metro_north_electric['time'])
LIRR_average_speed = 60 * sum(LIRR_electric['distance']) / sum(LIRR_electric['time'])
MBTA_average_speed = 60 * sum(MBTA_lines['distance']) / sum(MBTA_lines['time'])

with open('train_info.txt','w') as f:
	f.write('Metro North average speed: %0.2f \n' % metro_north_average_speed)
	f.write('LIRR average speed: %0.2f \n' % LIRR_average_speed)
	f.write('MBTA average speed: %0.2f \n' % MBTA_average_speed)
	f.write('Metro North average miles per stop: %0.2f \n' % metro_north_average_miles_per_stop)
	f.write('LIRR average miles per stop: %0.2f \n' % LIRR_average_miles_per_stop)
	f.write('MBTA average miles per stop: %0.2f \n' % MBTA_average_miles_per_stop)
	f.write('South Coast Rail average time savings due to electrification: %0.2f \n' % south_coast_rail_average_time_savings)
	f.write('Metro North cost per station: %0.3f \n' % time_cost_per_station(metro_north_electric))
	f.write('LIRR cost per station: %0.3f \n' % time_cost_per_station(LIRR_electric))
	f.write('MBTA cost per station: %0.3f \n' % time_cost_per_station(MBTA_lines))
	f.write('MBTA time savings per station due to electrification: %0.2f seconds \n' % (60 * MBTA_time_savings_per_station))
	f.write('MBTA average percent time savings due to electrification: %0.2f \n' % average_MBTA_percent_time_savings)
	
# average peak hour train time in minutes from corresponding municipality to Grand Central Terminal
inbound_train_times = [89.6, 40.97, 71.54, 101.43, 154, 125.25, 94.78, 55.56, 85.18, 66.18, 38.63, 46.38, 109, 75.37, 101.23, 48.83, 32.38, 61.3, 100.25, 94.25, 54.2, 35.14, 114.07, 69.67, 60.35, 93.64, 39.5, 42.52, 44.06, 50.86, 70.47, 84.19, 116.25, 60.17, 122.8, 55.41, 75.1, 55.06, 88.18, 100.4, 116.8, 65.93, 83.54, 71.75, 161, 63.6, 75.33, 132, 86, 142, 160, 64.33, 56.4, 80.33]

# track distance in miles from corresponding municpality to Grand Central Terminal
ny_distances = [48.5,22.3, 44.2, 69, 87.5, 82, 59, 33, 53.2, 41, 19, 24.1, 58.7,47.7, 73.5, 28.2, 15.1, 41.2, 63.7, 60.2,30.1, 16.6, 74, 41.2, 36.5, 63.2,20.5, 21.9, 22.2, 28.1, 43.7, 50.5, 76.5, 37.7, 64.9,38.4,52.5,32.4,55.4, 54,62.2,41.2,59, 38.9, 100.9, 37.2, 43.1,73.3,47.1, 89.3,90.1,27.9, 22.7,32.9]

# lat-long coordinates of train stations located at corresponding municipality
origin_coordinates_ny = [(41.196009, -73.432230),(41.032555, -73.774938),(41.119037, -73.371718),(41.271496, -72.963701),(41.554675, -73.046483),(41.814905, -73.562699),(41.194228, -73.131625),(41.047164, -73.542133),(41.410367, -73.621930),(41.096632, -73.421692),(40.989622, -73.808143),(40.985811, -73.682371),(41.324630, -73.434965),(41.337198, -73.657567),(41.706741, -73.937579),(41.109329, -73.795641),(40.910076, -73.810367),(41.285091, -73.930575),(41.564177, -73.601077),(41.511840, -73.604491),(41.157482, -73.868908),(40.911092, -73.783849),(41.297639, -72.927016),(41.146316, -73.496370),(41.209054, -73.728851),(41.222348, -73.060158),(40.953628, -73.736080),(41.039977, -73.872757),(40.969547, -73.712842),(41.021662, -73.625912),(41.295046, -73.676487),(41.143159, -73.257767),(41.742599, -73.575951),(41.076772, -73.471950),(41.395882, -73.450263),(41.246047, -73.921959),(41.405839, -73.932327),(41.157964, -73.774699),(41.177683, -73.187052),(41.266582, -73.441415),(41.376653, -73.416674),(41.259403, -73.684213),(41.506270, -73.984506),(40.700408, -73.324317),(40.964679, -72.194217),(40.852542, -73.411684),(40.736035, -73.207839),(40.919675, -72.666762),(40.855714, -73.200181),(40.894616, -72.389828),(41.066034, -72.427975),(40.858371, -73.620309),(40.590072, -73.664364),(40.875048, -73.532773)]

# average peak hour train time in minutes from corresponding municipality to South Station or North Station depending on location of municipality
boston_inbound_train_times = [74.2, 67.2, 54.2, 64, 59, 52.17, 46.17, 46.17, 35.92, 31.83, 25.44, 22, 83.33, 77.17, 69.17, 64.17, 57.14, 50.14, 45.8, 38.8, 34, 23.33, 66.2, 55, 48, 31.33, 25.33, 45.88, 37.88, 29.88, 24.88, 82.57, 72.33, 68.33, 59.17, 56.43, 49.27, 41.18, 34.71, 40.8, 59.33, 52.5, 45.57, 36.88, 32, 58, 48, 43, 38, 34, 59, 45, 36, 110.25, 95.25, 68.88, 50.22, 40.22, 31.88, 37.2, 29.2, 57.8, 57.8, 47.8, 36, 28, 28]

# track distance in miles from corresponding municpality to South Station/North Station
boston_distances = [35.3, 31.6, 25.4, 19.8, 36.2, 31.2, 27.6, 22.7, 22.7, 16.8, 12.8, 11.5, 49.6, 45.1, 39.4, 36.1, 30.1, 25.3, 20.1, 16.7, 13.7, 9.9, 32.9, 26.0, 22.8, 12, 9.9, 25.5, 21.8, 15.2, 12.7, 44.2, 36.4, 34.0, 27.4, 25.2, 21.4, 17.7, 13.5, 12.7, 27.5, 23, 19.1, 14.8, 12.5, 35.1, 28.1, 24.4, 21.2, 19.4, 27.6, 19.9, 16.2, 62.9, 51.9, 43.6, 31.8, 24.7, 17.9, 18.9, 15.6, 35.6, 35.6, 27.7, 20.0, 15, 15]

# lat-long coordinates of train stations located at corresponding municipality
origin_coordinates_boston_north = [(42.656264, -70.625835), (42.616724, -70.669326), (42.573798, -70.770282),(42.562296, -70.869211),(42.799960, -70.878720),(42.726879, -70.858968),(42.676984, -70.840446),(42.609749, -70.875638),(42.609749, -70.875638),(42.523930, -70.897178),(42.474026, -70.922849),(42.463281, -70.945053),(42.581713, -71.792407),(42.538802, -71.739633),(42.544908, -71.647973),(42.559714, -71.588730),(42.520479, -71.501511),(42.459944, -71.457985),(42.456773, -71.357391),(42.414110, -71.325716),(42.385960, -71.289040),(42.374552, -71.235660),(42.772681, -71.085992),(42.701948, -71.153386),(42.657990, -71.144848),(42.521963, -71.108433),(42.502186, -71.075370),(42.637440, -71.313802),(42.592521, -71.280141),(42.546851, -71.173779),(42.517138, -71.145395)]
origin_coordinates_boston_south = [(42.261808, -71.795073),(42.246447, -71.685074),(42.268806, -71.646168),(42.267501, -71.524082),(42.261910, -71.482721),(42.276158, -71.418815),(42.285981, -71.347086),(42.310176, -71.276603),(42.281058, -71.236823),(42.082981, -71.397372),(42.120457, -71.325336),(42.144208, -71.258151),(42.188861, -71.200080),(42.220739, -71.183424),(41.978690, -70.720944),(42.013457, -70.820810),(42.043262, -70.881571),(42.083171, -70.924248),(42.108094, -70.935691),(42.178722, -70.745799),(42.244516, -70.837633),(42.235951, -70.903496),(41.580124, -71.492414),(41.727877, -71.443351),(41.829098, -71.412900),(41.940778, -71.284514),(42.032722, -71.219757),(42.124989, -71.183567),(42.123822, -71.103032),(42.157071, -71.145546),(41.877954, -70.918784),(41.877954, -70.918784),(41.987760, -70.964822),(42.085731, -71.016343),(42.155443, -71.027618),(42.155443, -71.027618)]

# estimated decrease in Boston train times due to system-wide electrification 
boston_inbound_train_times_electrified = list(map(lambda x: (1 - (south_coast_rail_average_time_savings / 100)) * x, boston_inbound_train_times))

# (hr, minute) departure times to query google maps for driving trip times
morning_time_list = [(5,15), (5,45), (6,15), (6,45), (7,15), (7,45), (8,15), (8,45), (9,30), (10,30)]

# percentage of NY and Boston commuters who leave for work during corresponding time intervals 
morning_percentage_list = [0.024, 0.0341, 0.078, 0.0938, 0.1533, 0.1267, 0.152, 0.0845, 0.0804, 0.0313]
morning_percentage_list_boston = [0.0313, 0.0444, 0.0854, 0.1045, 0.1473, 0.1231, 0.1292, 0.0627, 0.0653, 0.0272]

# normalize percentages such that they sum to 1
morning_percentage_normalized = list(map(lambda x: x / sum(morning_percentage_list), morning_percentage_list))
morning_percentage_normalized_boston = list(map(lambda x: x / sum(morning_percentage_list_boston), morning_percentage_list_boston))

driving_dict = {}

for time in morning_time_list:
	driving_dict[time] = {'data': [], 'avg': ()}

# class to store transportation info corresponding to each municipality 
class city_info:
	def __init__(self, inbound_time, inbound_time_electrified, inbound_driving_data, inbound_driving_time, inbound_percent_diff, inbound_percent_diff_electrified):

		self.inbound_time = inbound_time
		self.inbound_time_electrified = inbound_time_electrified
		self.inbound_driving_data = deepcopy(inbound_driving_data)
		self.inbound_driving_time = inbound_driving_time
		self.inbound_percent_diff = inbound_percent_diff
		self.inbound_percent_diff_electrified = inbound_percent_diff_electrified

	def set_inbound_driving_time(self, time):
		self.inbound_driving_time = time

	def set_inbound_percent_diff(self, val):
		self.inbound_percent_diff = val

	def set_inbound_percent_diff_electrified(self, val):
		self.inbound_percent_diff_electrified = val

	def update_inbound_driving_data(self, key, val, field):
		if field == 'data':
			self.inbound_driving_data[key][field].append(val)
		elif field == 'avg':
			self.inbound_driving_data[key][field] = val
		else:
			return(print('invalid command'))

ny_city_dict = {}
boston_city_dict = {}

for i in range(len(ny_city_names)):
	ny_city_dict[ny_city_names[i]] = city_info(inbound_train_times[i], None, driving_dict, None, None, None)

for i in range(len(boston_city_names)):
	boston_city_dict[boston_city_names[i]] = city_info(boston_inbound_train_times[i], boston_inbound_train_times_electrified[i], driving_dict, None, None, None)

#gmaps = googlemaps.Client(key="AIzaSyAhUIf-ocfxHNh69bWhGvOHqPIZTmB2jZ4")
#gmaps = googlemaps.Client(key="AIzaSyDDg8okfqoJpFcV1pInFxWXZp1Bi_EHivg")
gmaps = googlemaps.Client(key="AIzaSyD5Rmf8cBDzdX3SkFu3Rh5FHIl7Sx2FiAs")

# lat-long coordinates of train destinations 
grand_central = (40.752053, -73.977412)
north_station = (42.365549, -71.064133)
south_station = (42.352620, -71.055425)

# predicted driving time from municipality to destination as a function of departure time and date
def commute_time(month, day, hour, minute, coordinates, dest):

	commute_results = gmaps.distance_matrix(origins = coordinates,
						destinations = dest,
						mode = 'driving',
						departure_time = datetime(2020, month, day, hour, minute, 0, 0),
						traffic_model = 'best_guess')
	try:
		for i in range(len(commute_results['rows'])):
			duration = commute_results['rows'][i]['elements'][0]['duration_in_traffic']['value']
			duration /= 60
			if dest == grand_central:
				ny_city_dict[ny_city_names[i]].update_inbound_driving_data((hour, minute), (round(duration,2)), 'data')
			else:
				offset = len(origin_coordinates_boston_north) if dest == south_station else 0
				boston_city_dict[boston_city_names[i + offset]].update_inbound_driving_data((hour, minute), (round(duration,2)), 'data')
	except KeyError:
		pass
	return(0)

# (month, day) dates in 2020 to use for Google Maps query (all dates are non-holliday weekdays)
chosen_days = [(1,7),(1,22),(2,6),(2,21),(3,9),(3,25),(4,9),(4,24),(5,12),(5,25),(6,24),(6,11),(7,17),(7,23),(8,3),(8,12),(9,10),(9,25),(10,18),(10,20),(11,12),(11,19),(12,4),(12,15)]#,(1,4)]#,(1,7),(1,9),(1,11),(1,15),(1,17),(1,21),(1,23),(1,25),(1,29),(1,31),(2,4),(2,6),(2,8),(2,12),(2,14),(2,18),(2,20),(2,22),(2,26),(2,28),(3,4),(3,6),(3,8),(3,12),(3,14),(3,18),(3,20),(3,22),(3,26),(3,28),(4,1),(4,3),(4,5),(4,9),(4,11),(4,15),(4,17),(4,19),(4,23),(4,25),(4,29),(5,1),(5,3),(5,7),(5,9),(5,13),(5,15),(5,17),(5,21),(5,23),(5,27),(5,29),(5,31),(6,4),(6,6),(6,10),(6,12),(6,14),(6,18),(6,20),(6,24),(6,26),(6,28),(7,2),(7,4),(7,8),(7,10),(7,12),(7,16),(7,18),(7,22),(7,24),(7,26),(7,30),(8,1),(8,5),(8,7),(8,9),(8,13),(8,15),(8,19),(8,21),(8,23),(8,27),(8,29),(9,2),(9,4),(9,6),(9,10),(9,12),(9,16),(9,18),(9,20),(9,24),(9,26),(9,30),(10,2),(10,4),(10,8),(10,10),(10,14),(10,16),(10,18),(10,22),(10,24),(10,28),(10,30),(11,1),(11,4),(11,6),(11,8),(11,12),(11,14),(11,18),(11,20),(11,22),(11,26),(11,28),(12,2),(12,4),(12,6),(12,10),(12,12),(12,16),(12,18),(12,20)]#,(12,24),(12,26),(12,30)]

# weighted average of driving time from all municipalities to Grand Central Terminal
def ny_execute():
	for (i,j) in chosen_days:
		for (hr, minute) in morning_time_list:
			commute_time(i, j, hr, minute, origin_coordinates_ny, grand_central)

	for city in ny_city_dict:
		for time in ny_city_dict[city].inbound_driving_data:
			data_list = ny_city_dict[city].inbound_driving_data[time]['data']
			if len(data_list) == 0:
				ny_city_dict[city].inbound_driving_data[time]['avg'] = 0
			else:
				ny_city_dict[city].inbound_driving_data[time]['avg'] = round((sum(data_list) / len(data_list)),4)

	for city in ny_city_dict:
		cnt = 0
		for i in range(len(morning_time_list)):
			cnt += (morning_percentage_normalized[i] * ny_city_dict[city].inbound_driving_data[morning_time_list[i]]['avg'])
		ny_city_dict[city].set_inbound_driving_time(round(cnt,4))
		ny_city_dict[city].set_inbound_percent_diff(round((100 * (((ny_city_dict[city].inbound_time) - cnt) / cnt)),4))  

	return(0)

# weighted average of driving time from all municipalities to South Station/North Station
def boston_execute():
	for (i,j) in chosen_days:
		for (hr, minute) in morning_time_list:
			commute_time(i, j, hr, minute, origin_coordinates_boston_north, north_station)
			commute_time(i, j, hr, minute, origin_coordinates_boston_south, south_station)

	for city in boston_city_dict:
		for time in boston_city_dict[city].inbound_driving_data:
			data_list = boston_city_dict[city].inbound_driving_data[time]['data']
			if len(data_list) == 0:
				boston_city_dict[city].inbound_driving_data[time]['avg'] = 0
			else:
				boston_city_dict[city].inbound_driving_data[time]['avg'] = round((sum(data_list) / len(data_list)),4)

	for city in boston_city_dict:
		cnt = 0
		for i in range(len(morning_time_list)):
			cnt += (morning_percentage_normalized_boston[i] * boston_city_dict[city].inbound_driving_data[morning_time_list[i]]['avg'])
		boston_city_dict[city].set_inbound_driving_time(round(cnt,4))
		boston_city_dict[city].set_inbound_percent_diff(round((100 * (((boston_city_dict[city].inbound_time) - cnt) / cnt)),4))  
		boston_city_dict[city].set_inbound_percent_diff_electrified(round((100 * (((boston_city_dict[city].inbound_time_electrified) - cnt) / cnt)),4))  

	return(0)

# plot relationship between time savings from taking the train instead of driving and the proportion of commuters that take the train
def ny_plot():

	# scatterplot with errorbars of time savings from taking commuter rail instead of driving vs proportion of commuters that use Metro North/LIRR
	time_diff = []
	for city in ny_city_dict:
		time_diff.append(ny_city_dict[city].inbound_percent_diff)
	time_diff = [-10.5629, -31.2047, -26.7565, -28.6007, 16.1664, -4.2034, -27.7132, -23.0832, -13.559, -28.0099, -30.5587, -25.8343, -0.4097, -11.0013, -15.1064, -20.6794, -40.0975, -27.2706, -11.537, -9.3869, -28.551, -35.0667, -20.7287, -14.8538, -16.8213, -31.442, -33.8803, -30.9548, -29.8966, -21.2894, -12.2459, -26.9518, -9.0314, -25.6519, 15.0554, -33.0758, -20.3175, -22.0113, -31.9376, -7.8168, 4.1335, -15.938, -18.5756, -24.7683, 2.533, -25.4984, -22.8318, 8.7722, -10.5198, 3.6509, 10.5088, -12.3398, -30.6877, -0.2544]
		
	PT_percent_array = np.array([percent_pt_ny])
	MOE_array = np.concatenate((np.array([[percent_pt_ny_MOE_lower]]), np.array([[percent_pt_ny_MOE_upper]])), axis = 0)
	time_diff_array = np.array([time_diff])
	plt.errorbar(time_diff_array, PT_percent_array, yerr = MOE_array, fmt = 'ro', ecolor = 'g', elinewidth = 1, capsize = 4)

	# add least-squares linear regression
	slope, intercept,_,_,_ = linregress(time_diff_array, PT_percent_array)

	axes = plt.gca()
	x_vals = np.array(axes.get_xlim())
	y_vals = intercept + slope * x_vals
	plt.plot(x_vals, y_vals, '--')
	plt.xlabel('time savings from taking train instead of driving (%)')
	plt.ylabel('proportion of commuters taking train')
	plt.title('NY: train usage as a function of time savings from train')

	with open('train_info.txt','a') as f:
		f.write('NY time difference: mean: %0.2f \n' % np.mean(time_diff_array))
		f.write('NY time difference: standard deviation: %0.2f \n' % np.std(time_diff_array))

	plt.savefig('errorplot_ny.png')
	plt.show()
	plt.clf()

	# plot after eliminating municipalities with high margin of errors (i.e. range of 90% confidence interval is larger than 0.4) that yield statistically insignificant data
	threshold = 0.4
	time_diff_f, percent_pt_ny_f, MOE_lower_f, MOE_upper_f = deepcopy(time_diff), deepcopy(percent_pt_ny), deepcopy(percent_pt_ny_MOE_lower), deepcopy(percent_pt_ny_MOE_upper)
	percent_pt_ny_f = [percent_pt_ny_f[i] for i,_ in enumerate(percent_pt_ny_f) if ny_con_int[i] < threshold]
	MOE_lower_f = [MOE_lower_f[i] for i,_ in enumerate(MOE_lower_f) if ny_con_int[i] < threshold]
	MOE_upper_f = [MOE_upper_f[i] for i,_ in enumerate(MOE_upper_f) if ny_con_int[i] < threshold]
	time_diff_f = [time_diff_f[i] for i,_ in enumerate(time_diff_f) if ny_con_int[i] < threshold]

	PT_percent_array_f =  np.array([percent_pt_ny_f])
	MOE_array_f = np.concatenate((np.array([[MOE_lower_f]]), np.array([[MOE_upper_f]])), axis = 0)
	time_diff_array_f = np.array([time_diff_f])
	plt.errorbar(time_diff_array_f, PT_percent_array_f, yerr = MOE_array_f, fmt = 'ro', ecolor = 'g', elinewidth = 1, capsize = 4)

	slope, intercept,_,_,_ = linregress(time_diff_array_f, PT_percent_array_f)

	axes = plt.gca()
	x_vals = np.array(axes.get_xlim())
	y_vals = intercept + slope * x_vals
	plt.plot(x_vals, y_vals, '--')
	plt.xlabel('time savings from taking train instead of driving (%)')
	plt.ylabel('proportion of commuters taking train')
	plt.title('NY: train usage as a function of time savings from train')
	plt.savefig('errorplot_ny_filtered.png')
	plt.show()
	plt.clf()

	# compute 95% confidence interval for correlation coefficient by drawing samples from multivariate normal distribution
	sample_num = 5000
	slope_list = [None] * sample_num
	lower_val = [None] * sample_num
	upper_val = [None] * sample_num

	for i in range(sample_num):
		pt_array_sampled = multivariate_Gaussian_sample(ny_pt_est, ny_pt_se)
		alone_array_sampled = multivariate_Gaussian_sample(ny_alone_est, ny_alone_se)
		carpool_array_sampled = multivariate_Gaussian_sample(ny_carpool_est, ny_carpool_se)
		other_sampled = multivariate_Gaussian_sample(ny_other_est, ny_other_se)
		PT_percent_array_sampled = pt_array_sampled / (pt_array_sampled + alone_array_sampled + carpool_array_sampled + other_sampled)
		PT_percent_array_sampled_f = [PT_percent_array_sampled[i] for i,_ in enumerate(PT_percent_array_sampled) if ny_con_int[i] < threshold]
		X = sm.add_constant(time_diff_f)
		est = sm.OLS(PT_percent_array_sampled_f, X)
		est2 = est.fit()
		slope_list[i] = est2.params[1]
		lower_val[i] = est2.conf_int(alpha=0.05, cols=None)[1][0]
		upper_val[i] = est2.conf_int(alpha=0.05, cols=None)[1][1]

	with open('train_info.txt','a') as f:
		f.write('NY filtered linear regression: slope: %0.5f \n' % slope)
		f.write('NY filtered linear regression: slope 95 percent confidence interval: [%0.5f, %0.5f] \n' % (np.mean(lower_val), np.mean(upper_val)))

	return(0)

# plot relationship between time savings from taking the train instead of driving and the proportion of commuters that use the MBTA commuter rail system.
def boston_plot():

	# scatterplot with errorbars of time savings from taking commuter rail instead of driving vs proportion of commuters who take commuter rail 
	time_diff = []
	time_diff_electrified = []
	for city in boston_city_dict:
		time_diff.append(boston_city_dict[city].inbound_percent_diff)
		time_diff_electrified.append(boston_city_dict[city].inbound_percent_diff_electrified)

	time_diff = [5.2141, 5.442, -10.6562, 17.633, -7.6022, -22.6794, -27.5066, -19.9266, -37.7033, -42.4573, -44.5762, -45.3874, -6.6513, -10.3988, -13.4741, -19.6215, -16.6433, -26.4944, -0.8501, -7.3174, -11.9754, -18.2945, -2.7668, -7.5414, -11.2952, -23.6327, -39.7092, -25.7704, -34.0405, -32.3867, -41.414, 12.5249, -0.9406, 3.1764, 4.1588, -1.1603, -0.0026, -8.3597, 9.5363, 9.3397, -20.0184, -21.9599, -25.2431, -32.2491, -28.874, -17.5439, -35.2396, -39.1786, -40.776, -43.1745, -11.8167, -28.9721, -25.8407, 6.3327, 2.2936, -19.3834, -33.3423, -39.6964, -42.812, -32.6387, -43.1651, -22.2328, -22.2328, -35.4953, -41.7688, -45.4142, -45.4142]
		
	PT_percent_array = np.array([percent_pt_boston])
	MOE_array = np.concatenate((np.array([[percent_pt_boston_MOE_lower]]), np.array([[percent_pt_boston_MOE_upper]])), axis = 0)
	time_diff_array = np.array([time_diff])
	plt.errorbar(time_diff_array, PT_percent_array, yerr = MOE_array, fmt = 'ro', ecolor = 'g', elinewidth = 1, capsize = 4)

	# add least-squares linear regression
	slope, intercept,_,_,_ = linregress(time_diff_array, PT_percent_array)

	axes = plt.gca()
	x_vals = np.array(axes.get_xlim())
	y_vals = intercept + slope * x_vals
	plt.plot(x_vals, y_vals, '--')

	plt.xlabel('time savings from taking train instead of driving (%)')
	plt.ylabel('proportion of commuters taking train')
	plt.title('Boston: train usage as a function of time savings from train')

	plt.savefig('errorplot_boston.png')
	plt.show()
	plt.clf()

	# compute 95% confidence interval for correlation coefficient by drawing samples from multivariate normal distribution
	sample_num = 5000
	slope_list = [None] * sample_num
	lower_val = [None] * sample_num
	upper_val = [None] * sample_num
	for i in range(sample_num):
		pt_array_sampled = multivariate_Gaussian_sample(boston_pt_est, boston_pt_se)
		alone_array_sampled = multivariate_Gaussian_sample(boston_alone_est, boston_alone_se)
		carpool_array_sampled = multivariate_Gaussian_sample(boston_carpool_est, boston_carpool_se)
		other_sampled = multivariate_Gaussian_sample(boston_other_est, boston_other_se)
		PT_percent_array_sampled = pt_array_sampled / (pt_array_sampled + alone_array_sampled + carpool_array_sampled + other_sampled)
		X = sm.add_constant(time_diff)
		est = sm.OLS(PT_percent_array_sampled, X)
		est2 = est.fit()
		slope_list[i] = est2.params[1]
		lower_val[i] = est2.conf_int(alpha=0.05, cols=None)[1][0]
		upper_val[i] = est2.conf_int(alpha=0.05, cols=None)[1][1]

	with open('train_info.txt','a') as f:
		f.write('Boston time difference: mean: %0.2f \n' % np.mean(time_diff_array))
		f.write('Boston time difference: standard deviation: %0.2f \n' % np.std(time_diff_array))
		f.write('Boston post-electrification time difference: mean: %0.2f \n' % np.mean(time_diff_electrified))
		f.write('Boston post-electrification time difference: standard deviation: %0.2f \n' % np.std(time_diff_electrified))
		f.write('Boston linear regression: slope: %0.5f \n' % slope)
		f.write('Boston linear regression: slope 99 percent confidence interval: [%0.5f, %0.5f] \n' % (slope_min, slope_max))
		f.write('Boston linear regression: slope 95 percent confidence interval: [%0.5f, %0.5f] \n' % (np.mean(lower_val), np.mean(upper_val)))

	return(0)

ny_execute()
ny_plot()	

boston_execute()
boston_plot()


