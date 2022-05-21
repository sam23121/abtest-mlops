import json
import pandas as pd

class LoadData:
    """
    this class contains functions to load data from json, excel or csv files into a pandas dataframe 
    
    Return
    ------
    dataframe
    """
    
    def __init__(self):
        pass
    
    def read_json(self, json_file: str, dfExtractFunc: object )->pd.DataFrame:
        """
        json file reader to open and read json files into a panda dataframe
        Args:
        -----
        json_file: str - path of a json file
        dfExtractFunc: object - The function that will used to extract data from the list formed from the json file
        Returns
        -------
        dataframe containing data extracted from the json file
        """

        data_list = []
        for item in open(json_file,'r'):
            data_list.append(json.loads(item))

        df = dfExtractFunc(data_list)
        return df
    
    def read_excel(self, excel_file, startRow=0)->pd.DataFrame: 
        """
        excel file reader to open and read excel files into a panda dataframe
        Args:
        -----
        excel_file: str - path of a excel file
        engine: str - sets the default engine used by pandas to load the excel file
        startRow: int - sets the first row in excel sheet where pandas should start loading from
        Returns
        -------
        dataframe containing data extracted from the excel file
        """
        
        df = pd.read_excel(excel_file,engine='openpyxl')
        return df
    
    def read_csv(self, csv_file)->pd.DataFrame:
        """
        csv file reader to open and read csv files into a panda dataframe
        Args:
        -----
        csv_file: str - path of a json file
        Returns
        -------
        dataframe containing data extracted from the csv file        
        
        """
        
        return pd.read_csv(csv_file)