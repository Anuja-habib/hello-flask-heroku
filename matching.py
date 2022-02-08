import sys
sys.path.insert(1, '../0_data')
sys.path.insert(1, '../1_config')
sys.path.insert(1, '../')

#from queryrunner_client import Client
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import re

import unicodedata
import geopy.distance
#import abydos.distance

import requests
import json
from math import floor, ceil

from fuzzywuzzy import fuzz
from fastDamerauLevenshtein import damerauLevenshtein
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import pipeline

import pickle
import logging

from tqdm.notebook import tqdm, trange
import warnings
warnings.filterwarnings('ignore')
import logging
import sys

# logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO, stream=sys.stdout)

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

stopwords = []

#############################################################################
############# Compute Distance Between Latitude and Longitude  ##############
#############################################################################
    
def calculate_distance(lat1, long1, lat2, long2):

    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(long2-long1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    
    return d*1000
#############################################################################
######################### cleaninig  ########################################
#############################################################################

def clean_df(df):
    print('df',df)
    df.replace('\\N', np.nan, inplace = True)
    print('df.replace',df)
    df.dropna(inplace = True)
    print('df.dropna',df)
    df.drop_duplicates(subset = 'id', keep='first', inplace = True) # Dataset are sorted by descending date - keep most recent data
    print('df.drop_duplicates',df)
    #df['name'] = df.name.map(lambda x: clean_names(x))
    #print('df.name.map',df)
    df[['latitude', 'longitude']] = df[['latitude', 'longitude']].apply(pd.to_numeric)
    print('latitude.longitude',df)
    return df

#############################################################################
#########################    stopwords generation    ########################
#############################################################################

def tfidf_stopwords(df1, df2):
    print('df1',df1)
    print('df2',df2)
    cleanname1=df1.name.str
    print(cleanname1)
    fraction = 0.1 # High to avoid this to be used
    #print('fraction',fraction)
    chains_1 = cleanname1.lower().replace("[\(\[].*?[\)\]]", "",regex=True).value_counts().loc[lambda x: x>=len(df1)*fraction/2].index.tolist()
    #print('chains_1',chains_1)
    chains_2 = df2.name.str.lower().replace("[\(\[].*?[\)\]]", "",regex=True).value_counts().loc[lambda x: x>=len(df2)*fraction/2].index.tolist() 
    #print('chains_2',chains_2)
    chains = chains_1 + chains_2
    #print('chains',chains)
    corpus = list(df1.name.values) + list(df2.name.values)
    #print('corpus',corpus)
    cv = CountVectorizer()#corpus,max_df=fraction, lowcase=True, ngram_range=(1, 1))
    #print('cv',cv)
    count_vector = cv.fit_transform(corpus)
    #print('count_vector',count_vector)
    stopwords = [stopword for stopword in list(cv.stop_words_) if not any(stopword in chain for chain in chains)]
    #print('stopwords',stopwords)
    
    stopwords_pre = './1_config/stopwords.txt'   # PREDEFINED ENGLISH STOPWORDS
    print('stopwords_pre',stopwords_pre)
    with open(stopwords_pre, 'r') as file:
        #print('open(stopwords_pre')
        english_stopwords = list(file.read().splitlines())
        #print('english_stopwords',english_stopwords)
    #print(set(stopwords + english_stopwords + ["noodle","food"]))   
    return set(stopwords + english_stopwords + ["noodle","food"])
#############################################################################
######################### String Similarity Metrics  ########################
#############################################################################

def jaccard(name1, name2):
    name1_clean = name1.lower()
    list1 = re.sub('\'', '', name1_clean).split()

    name2_clean = name2.lower()
    list2 = re.sub('\'', '', name2_clean).split()

    filtered_words1 = set(list1) - set(stopwords)
    filtered_words2 = set(list2) - set(stopwords)

    try:
        return len(set(filtered_words1).intersection(set(filtered_words2))) / len(set(filtered_words1).union(set(filtered_words2)))
    except ZeroDivisionError:
        try:
            return len(set(list1).intersection(set(list2))) / len(set(list1).union(set(list2)))
        except ZeroDivisionError:
            return 0

        
def jaccard_latin(name1, name2):
    
    name1_clean = name1.lower()
    name1_clean = re.sub('[^a-z0-9 \']+', ' ', name1_clean)
    if len(name1_clean) == 1: #Only a space left
        list1 = re.sub('\'', '', name1.lower()).split()
    else:
        list1 = re.sub('\'', '', name1_clean).split()

    name2_clean = name2.lower()
    name2_clean = re.sub('[^a-z0-9 \']+', ' ', name2_clean)
    if len(name2_clean) == 1: #Only a space left
        list2 = re.sub('\'', '', name2.lower()).split()
    else:
        list2 = re.sub('\'', '', name2_clean).split()
    
    #list2 = re.sub('\'', '', name2_clean).split()

    filtered_words1 = set(list1) - set(stopwords)
    filtered_words2 = set(list2) - set(stopwords)

    try:
        return len(set(filtered_words1).intersection(set(filtered_words2))) / len(set(filtered_words1).union(set(filtered_words2)))
    except ZeroDivisionError:
        try:
            return len(set(list1).intersection(set(list2))) / len(set(list1).union(set(list2)))
        except ZeroDivisionError:
            return 0         


def levenshtein_fuzzy_token_set_ratio(name1, name2):
    name1_clean = name1.lower()
    name2_clean = name2.lower()

    name1_clean = name1_clean.split()
    name1_clean  = [word for word in name1_clean if word.lower() not in stopwords]
    name1_clean = ' '.join(name1_clean)
    if len(name1_clean) == 0:
        name1_clean = name1.lower().strip()

    name2_clean = name2_clean.split()
    name2_clean  = [word for word in name2_clean if word.lower() not in stopwords]
    name2_clean = ' '.join(name2_clean)
    if len(name2_clean) == 0:
        name2_clean = name2.lower().strip()

    return  fuzz.token_set_ratio(name1_clean, name2_clean)

def levenshtein_damerau(name1, name2):
    name1_clean = name1.lower()
    name2_clean = name2.lower()

    name1_clean = name1_clean.split()
    name1_clean  = [word for word in name1_clean if word.lower() not in stopwords]
    name1_clean = ' '.join(name1_clean)
    if len(name1_clean) == 0:
        name1_clean = name1.lower().strip()

    name2_clean = name2_clean.split()
    name2_clean  = [word for word in name2_clean if word.lower() not in stopwords]
    name2_clean = ' '.join(name2_clean)
    if len(name2_clean) == 0:
        name2_clean = name2.lower().strip()

    return damerauLevenshtein(name1_clean, name2_clean, similarity=True)



# Function to calculate the Jaro Similarity of two strings
def jaro_distance(s1, s2):
    # If the s are equal
    if (s1 == s2):
        return 1.0

    # Length of two s
    len1 = len(s1)
    len2 = len(s2)

    # Maximum distance upto which matching is allowed
    max_dist = floor(max(len1, len2) / 2) - 1
    match = 0

    # Hash for matches
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)

    # Traverse through the first
    for i in range(len1):

        # Check if there is any matches
        for j in range(max(0, i - max_dist), min(len2, i + max_dist + 1)):

            # If there is a match
            if (s1[i] == s2[j] and hash_s2[j] == 0):
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break

    # If there is no match
    if (match == 0):
        return 0.0

    # Number of transpositions
    t = 0
    point = 0

    # Count number of occurances where two characters match but there is a
    # third matched character in between the indices
    for i in range(len1):
        if (hash_s1[i]):

            # Find the next matched character in second
            while (hash_s2[point] == 0):
                point += 1

            if (s1[i] != s2[point]):
                point += 1
                t += 1
    t = t//2

    # Return the Jaro Similarity
    return (match/ len1 + match / len2 + (match - t + 1) / match)/ 3.0



def monge_elkan(name1, name2):
    name1_clean = name1.lower()
    name2_clean = name2.lower()

    cummax = 0
    for ws in name1.split(" "):
        maxscore=0
        for wt in name2.split(" "):
            maxscore = max(maxscore,jaro_distance(ws,wt))
        cummax += maxscore
    return  cummax/len(name1.split(" "))



def monge_elkan_jaro(name1, name2):
    # Monge elkan is asymetric
    name1_clean = name1.lower()
    name2_clean = name2.lower()

    name1_clean = name1_clean.split()
    name1_clean  = [word for word in name1_clean if word.lower() not in stopwords]
    name1_clean = ' '.join(name1_clean)
    if len(name1_clean) == 0:
        name1_clean = name1.lower().strip()

    name2_clean = name2_clean.split()
    name2_clean  = [word for word in name2_clean if word.lower() not in stopwords]
    name2_clean = ' '.join(name2_clean)
    if len(name2_clean) == 0:
        name2_clean = name2.lower().strip()

    score1 = monge_elkan(name1_clean, name2_clean)
    score2 = monge_elkan(name2_clean, name1_clean)
    return (score1 + score2) / 2



def compute_metrics(dd, latin, geo_info=True):
    
    logging.info('Computing Jaccard Similarity')
    
    if latin:
        dd['jaccard'] = np.vectorize(jaccard_latin)(dd.comp1_name.values, dd.comp2_name.values)
    else:
        dd['jaccard'] = np.vectorize(jaccard)(dd.comp1_name.values, dd.comp2_name.values)  
    logging.info('Computing Monge Elkan Similarity')
    
    dd['monge_elkan_jiro_winkler'] = np.vectorize(monge_elkan_jaro)(dd.comp1_name.values, dd.comp2_name.values)
    logging.info('Computing fuzzy Levenshtein') 
    
    dd['levenshtein_fuzzy_token_set_ratio'] = np.vectorize(levenshtein_fuzzy_token_set_ratio)(dd.comp1_name.values, dd.comp2_name.values)
    logging.info('Computing Levenshtein Damerau')
    
    dd['levenshtein_damerau'] = np.vectorize(levenshtein_damerau)(dd.comp1_name.values, dd.comp2_name.values)
    logging.info('Finishing up...')
    
    dd['combined'] = dd['jaccard'] + dd['levenshtein_fuzzy_token_set_ratio'] + dd['levenshtein_damerau'] + dd['monge_elkan_jiro_winkler']
    
    dd['weighted_combined'] = ((3*dd['jaccard']) + 2*dd['monge_elkan_jiro_winkler'] + dd['levenshtein_fuzzy_token_set_ratio'] + dd['levenshtein_damerau'])/7
    
    if geo_info:
        dd['distance_meters'] = np.vectorize(calculate_distance)(dd.comp1_latitude.values, dd.comp1_longitude.values, dd.comp2_latitude.values, dd.comp2_longitude.values)
    
    dd['diff_length_names'] = np.vectorize(lambda x,y : abs(len(x) - len(y)))(dd.comp1_name.values, dd.comp2_name.values)
    
    return dd


def doMatching(dataset1, dataset2, stopw, distance_range, latin):
    logging.info('Starting Matching Process') 
    
    global stopwords
    stopwords = stopw
    
    # Pick the right order
    if len(dataset1) > len(dataset2):
        comp1 = dataset1.copy()
        comp2 = dataset2.copy()
        tracker = True
    else:
        comp1 = dataset2.copy()
        comp2 = dataset1.copy()
        tracker = False

    comp1.columns = ['comp1_id', 'comp1_name', 'comp1_latitude', 'comp1_longitude']
    comp2.columns = ['comp2_id', 'comp2_name', 'comp2_latitude', 'comp2_longitude']


    # Create resto matching boundary with lat/longs in dataset2
    comp2['lower_lat'] = comp2.comp2_latitude.map(lambda x: x - distance_range)
    comp2['upper_lat'] = comp2.comp2_latitude.map(lambda x: x + distance_range)
    comp2['lower_long'] = comp2.comp2_longitude.map(lambda x: x - distance_range)
    comp2['upper_long'] = comp2.comp2_longitude.map(lambda x: x + distance_range)
    

    # Order lat/longs in dataset1 to generate lower and upper lat/long indices
    complat = comp1.sort_values(by = 'comp1_latitude')
    complong = comp1.sort_values(by = 'comp1_longitude')
    #print('complat',complat)
    #print('complong',complong)
    lower_indices_lat = np.searchsorted(complat.comp1_latitude.values, comp2.lower_lat, side='left')
    upper_indices_lat = np.searchsorted(complat.comp1_latitude.values, comp2.upper_lat, side='right')
    #print('lower_indices_lat',lower_indices_lat)
    #print('upper_indices_lat',upper_indices_lat)
    lower_indices_long = np.searchsorted(complong.comp1_longitude.values, comp2.lower_long, side= 'left')
    upper_indices_long = np.searchsorted(complong.comp1_longitude.values, comp2.upper_long, side = 'right')
    #print('lower_indices_long',lower_indices_long)
    #print('upper_indices_long',upper_indices_long)
    complat_ids = list(complat.comp1_id)
    complong_ids = list(complong.comp1_id)
    #print('complat_ids',complat_ids)
    #print('complong_ids',complong_ids)

    # Create iterator to find near ids
    comp2_iterator = zip(comp2.comp2_id.values, comp2.comp2_name.values,
                         comp2.comp2_latitude.values, comp2.comp2_longitude.values,
                         lower_indices_lat, upper_indices_lat,
                         lower_indices_long, upper_indices_long)

    logging.info('Create Matching Candidates')
    nearby_ids = []
    for comp2_id, comp2_name, comp2_latitude, comp2_longitude, lower_lat, upper_lat, lower_long, upper_long in comp2_iterator:

        lat_ids = complat_ids[lower_lat:upper_lat]
        long_ids = complong_ids[lower_long:upper_long]
        intersection = list(set(lat_ids) & set(long_ids))
        ids = [(comp2_id, comp2_name, comp2_latitude, comp2_longitude, comp1_id) for comp1_id in intersection]

        nearby_ids.extend(ids)

    #print('nearby_ids',nearby_ids)
    search_results = pd.DataFrame(nearby_ids, columns = ['comp2_id', 'comp2_name', 'comp2_latitude', 'comp2_longitude', 'comp1_id'])
    #print('search_results',search_results)
    latlong_filtered = pd.merge(search_results, comp1, how = 'inner', on = 'comp1_id')
    print('latlong_filtered!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',latlong_filtered)
    latlong_filtered = compute_metrics(latlong_filtered, latin=latin)
    return latlong_filtered




