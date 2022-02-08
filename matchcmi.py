from cgi import print_directory
from flask import Flask
import psycopg2
import pandas as pd
import matching
import crud
import config

app = Flask(__name__)




@app.route('/')
def match():
    distance_range  = 0.0001 #~250M
    LATIN           = True 
    try:
        conn = psycopg2.connect(config.read_db_config)
    except Exception:
        resultJson ={"Warning ": "There was problem in connecting to the database !"}      
        return resultJson    
    if (conn):
        cursor = conn.cursor()
        conn.autocommit = False
        try:
                cursor.execute('''SELECT Id,Name,latitude,longitude from Ubersdata ''')
                result1 = cursor.fetchall()
                cursor.execute('''SELECT Id,Name,latitude,longitude from compdata  ''')
                result2 = cursor.fetchall() 
        except Exception:
              resultJson ={"Warning ": "There was problem in fetching data from database !"}      
              return resultJson  
        print(type(result1))
        print(len(result1))
        
        if len(result1) > 0 & len(result2) > 0:
                df = pd.DataFrame(result1, 
                                columns = ['id','name','latitude','longitude'])
                #print(df)
                
                df1 = pd.DataFrame(result2, 
                                columns = ['id','name','latitude','longitude'])
                #print(df1)
                
                dataset1 = df[["id", "name", "latitude", "longitude"]] 
                #print(dataset1)
                dataset2 = df1[["id", "name", "latitude", "longitude"]]
                #print(dataset2)
                #cursor = conn.cursor()
                dataset1 = matching.clean_df(dataset1)
                dataset2 = matching.clean_df(dataset2)
                #print(dataset1)
                #print("---------------------------------------------------")
                #print(dataset2)
                stopwords = matching.tfidf_stopwords(dataset1, dataset2)
                tracker = True if len(dataset1) > len(dataset2) else False
                #print(tracker)
                latlong_filtered = matching.doMatching(dataset1, dataset2, stopwords, distance_range, latin=LATIN)
                latlong_filtered = latlong_filtered[latlong_filtered.weighted_combined > 0.06]
                #print(latlong_filtered)
                lfb = latlong_filtered.copy(deep=True)
                lfb["Evaluation"] = "NOT SURE"
                lfb.loc[lfb['jaccard']>=0.4, 'Evaluation'] = "TRUE"

                lfb.loc[(lfb.Evaluation == "NOT SURE") 
                        & (lfb.levenshtein_damerau > 0.60) 
                        & (lfb.levenshtein_fuzzy_token_set_ratio > 80), "Evaluation"] = "TRUE"

                lfb.loc[(lfb.Evaluation == "NOT SURE") 
                        & (lfb.levenshtein_damerau > 0.70), 'Evaluation'] = "TRUE"

                lfb.loc[(lfb.Evaluation == "NOT SURE") 
                        &  (lfb.levenshtein_fuzzy_token_set_ratio <= 20) 
                        & (lfb.levenshtein_damerau < 0.2), 'Evaluation'] = "FALSE"

                lfb.loc[(lfb.Evaluation == "NOT SURE") 
                        & (lfb.levenshtein_fuzzy_token_set_ratio == 100) 
                        & (lfb.monge_elkan_jiro_winkler > 0.9), 'Evaluation'] = "TRUE"

                lfb.loc[(lfb.Evaluation == "NOT SURE") 
                        & (lfb.jaccard >= 0.34) 
                        & (lfb.distance_meters <= 60), 'Evaluation'] = "TRUE"

                lfb.loc[(lfb.Evaluation == "NOT SURE") 
                        & (lfb.jaccard >= 0.3) 
                        & (lfb.distance_meters <= 10), 'Evaluation'] = "TRUE"

                lfb.loc[(lfb.Evaluation == "NOT SURE") 
                        & (lfb.levenshtein_fuzzy_token_set_ratio == 100) 
                        & (lfb.distance_meters <= 50), 'Evaluation'] = "TRUE"
                #print('After lfb')
                #print('lfb.Evaluation',lfb.Evaluation)
                #print('lfb.levenshtein_fuzzy_token_set_ratio',lfb.levenshtein_fuzzy_token_set_ratio)
                #print('lfb.distance_meters',lfb.distance_meters)
                #print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                lfb_2 = lfb[lfb.Evaluation == "TRUE"].sort_values(by = ['jaccard', 'levenshtein_fuzzy_token_set_ratio','distance_meters'], ascending = [False, False, True])
                #print('lfb_2',lfb_2)
                results = lfb_2.drop_duplicates(subset = 'comp1_id').drop_duplicates(subset = 'comp2_id')
                #print('results',results)
                print("% MATCHED: ", len(results)/len(dataset2))
                print("ABSOLUTE MATCHED: ", len(results))

                if tracker:
                        #print('tracker',tracker)
                        on_eats = results.merge(dataset1[["id"]], how="left", left_on="comp1_id", right_on="id")
                        #print('on_eats',on_eats)
                        df_final_competitor = dataset2.merge(on_eats[["comp2_id", "comp1_id"]], how='left', left_on=['id'], right_on=['comp2_id'])
                        #print('df_final_competitor',df_final_competitor)
                        df_final_competitor = df_final_competitor[list(dataset2.columns)+["comp1_id"]]
                        print('df_final_competitor final',df_final_competitor)
                else:
                        #print('tracker',tracker)
                        on_eats = results.merge(dataset1[["id"]], how="left", left_on="comp2_id", right_on="id")
                        #print('on_eats',on_eats)
                        df_final_competitor = dataset2.merge(on_eats[["comp2_id", "comp1_id"]], how='left', left_on=['id'], right_on=['comp1_id'])
                        #print('df_final_competitor',df_final_competitor)
                        df_final_competitor = df_final_competitor[list(dataset2.columns)+["comp2_id"]]
                        print('df_final_competitor final',df_final_competitor)
                resultJson = df_final_competitor.to_json(orient ='records')
        
                print('resultJson',resultJson)
               #save = crud.commitToDatabase(cursor,resultJson)
                return resultJson
                
        else :
                resultJson ={"Warning ": "Empty dataset provided !"}
        

                return resultJson
    
if __name__ == '__main__':
    app.run()
