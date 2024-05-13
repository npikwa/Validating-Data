# Validating-Data 

In this Project I'll be diving into an agricultural dataset to validate the data. I'll build a data pipeline that will ingest and clean the data, cleaning up my code significantly. Once that’s ready, I’ll complete the data validation.

## The plan:
1. Create a null hypothesis.
2. Import the `MD_agric_df` dataset and clean it up.
3. Import the weather data.
4. Map the weather data to the field data.
5. Calculate the means of the weather station dataset and the means of the main dataset.
6. Calculate all the parameters we need to do a t-test. 
7. Interpret our results.

## Data dictionary
**1. Geographic features**

- **Field_ID:** A unique identifier for each field (BigInt).
 
- **Elevation:** The elevation of the field above sea level in metres (Float).

- **Latitude:** Geographical latitude of the field in degrees (Float).

- **Longitude:** Geographical longitude of the field in degrees (Float).

- **Location:** Province the field is in (Text).

- **Slope:** The slope of the land in the field (Float).

**2. Weather features**

- **Field_ID:** Corresponding field identifier (BigInt).

- **Rainfall:** Amount of rainfall in the area in mm (Float).

- **Min_temperature_C:** Average minimum temperature recorded in Celsius (Float).

- **Max_temperature_C:** Average maximum temperature recorded in Celsius (Float).

- **Ave_temps:** Average temperature in Celcius (Float).

**3. Soil and crop features**

- **Field_ID:** Corresponding field identifier (BigInt).

- **Soil_fertility:** A measure of soil fertility where 0 is infertile soil, and 1 is very fertile soil (Float).

- **Soil_type:** Type of soil present in the field (Text).

- **pH:** pH level of the soil, which is a measure of how acidic/basic the soil is (Float).

**4. Farm management features**

- **Field_ID:** Corresponding field identifier (BigInt).

- **Pollution_level:** Level of pollution in the area where 0 is unpolluted and 1 is very polluted (Float).

- **Plot_size:** Size of the plot in the field (Ha) (Float).

- **Chosen_crop:** Type of crop chosen for cultivation (Text).

- **Annual_yield:** Annual yield from the field (Float). This is the total output of the field. The field size and type of crop will affect the Annual Yield

- **Standard_yield:** Standardised yield expected from the field, normalised per crop (Float). This is independent of field size, or crop type. Multiplying this number by the field size, and average crop yield will give the Annual_Yield.

<br>

**Weather_station_data (CSV)**

- **Weather_station_ID:** The weather station the data originated from. (Int)

- **Message:** The weather data was captured by sensors at the stations, in the format of text messages.(Str)

**Weather_data_field_mapping (CSV)**

- **Field_ID:** The id of the field that is connected to a weather station. This is the key we can use to join the weather station ID to the original data. (Int)

- **Weather_station_ID:** The weather station that is connected to a field. If a field has `weather_station_ID = 0` then that field is closest to weather station 0. (Int)

<br>

## This is executed using Jupyter Notebook

## Cleaning up the data pipeline

    import pandas as pd # importing the Pandas package with an alias, pd
    from sqlalchemy import create_engine, text # Importing the SQL interface. If this fails, run !pip install sqlalchemy in another cell.
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create an engine for the database
    engine = create_engine ('sqlite:///Maji_Ndogo_farm_survey_small.db') # Make sure to have the .db file in the same directory as this notebook, and the file name matches.
<br>

    sql_query = """
    SELECT *
    FROM geographic_features
    LEFT JOIN weather_features USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
    """

    #Create a connection object
    with engine.connect() as connection:
    
    # Use Pandas to execute the query and store the result in a DataFrame
    MD_agric_df = pd.read_sql_query(text(sql_query), connection)
<br>

    MD_agric_df.rename(columns={'Annual_yield': 'Crop_type_Temp', 'Crop_type': 'Annual_yield'}, inplace=True)
    MD_agric_df.rename(columns={'Crop_type_Temp': 'Crop_type'}, inplace=True)
    MD_agric_df['Elevation'] = MD_agric_df['Elevation'].abs()

    # Correcting 'Crop_type' column
    def correct_crop_type(crop):
            crop = crop.strip()  # Remove trailing spaces
            corrections = {'cassaval': 'cassava','wheatn': 'wheat','teaa': 'tea'}
        return corrections.get(crop, crop)  # Get the corrected crop type, or return the original if not in corrections

    # Apply the correction function to the Crop_type column
    MD_agric_df['Crop_type'] = MD_agric_df['Crop_type'].apply(correct_crop_type)
<br>

    weather_station_df = pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_station_data.csv")
    weather_station_mapping_df = pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv")
<br>

    import re # Importing the regex pattern
    import numpy as np


    patterns = {'Rainfall': r'(\d+(\.\d+)?)\s?mm','Temperature': r'(\d+(\.\d+)?)\s?C','Pollution_level': r'=\s*(-?\d+(\.\d+)?)|Pollution at \s*(-?\d+(\.\d+)?)'
    }

    def extract_measurement(message):
        """
        Extracts a numeric measurement value from a given message string.

        The function applies regular expressions to identify and extract numeric values related to different types of measurements such as
        Rainfall, Average Temperatures, and Pollution Levels from a text message.
        It returns the key of the matching record, and first matching value as a floating-point number.
    
        Parameters:
        message (str): A string message containing the measurement information.

        Returns:
        float: The extracted numeric value of the measurement if a match is found;
        otherwise, None.

        The function uses the following patterns for extraction:
        - Rainfall: Matches numbers (including decimal) followed by 'mm', optionally spaced.
        - Ave_temps: Matches numbers (including decimal) followed by 'C', optionally spaced.
        - Pollution_level: Matches numbers (including decimal) following 'Pollution at' or '='.
    
        Example usage:
        extract_measurement("【2022-01-04 21:47:48】温度感应: 现在温度是 12.82C.")
        # Returns: 'Temperature', 12.82
        """
    
        for key, pattern in patterns.items(): # Loop through all of the patterns and check if it matches the pattern value.
            match = re.search(pattern, message)
            if match:
                # Extract the first group that matches, which should be the measurement value if all previous matches are empty.
                # print(match.groups()) # Uncomment this line to help you debug your regex patterns.
                return key, float(next((x for x in match.groups() if x is not None)))
    
        return None, None

    # The function creates a tuple with the measurement type and value into a Pandas Series
      result = weather_station_df['Message'].apply(extract_measurement)

    # Create separate columns for 'Measurement' and 'extracted_value' by unpacking the tuple with Lambda functions.
    weather_station_df['Measurement'] = result.apply(lambda x: x[0])
    weather_station_df['Value'] = result.apply(lambda x: x[1])
<br>

    # The function creates a tuple with the measurement type and value into a Pandas Series
    result = weather_station_df['Message'].apply(extract_measurement)

    # Create separate columns for 'Measurement' and 'extracted_value' by unpacking the tuple with Lambda functions.
    weather_station_df['Measurement'] = result.apply(lambda x: x[0])
    weather_station_df['Value'] = result.apply(lambda x: x[1])

    weather_station_means = weather_station_df.groupby(by = ['Weather_station_ID','Measurement'])['Value'].mean(numeric_only = True)
    weather_station_means = weather_station_means.unstack()
    weather_station_means
<br>

    # Use this line of code to see which messages are not assigned yet.
    weather_station_df[(weather_station_df['Measurement'] == None)|(weather_station_df['Value'].isna())]
<br>

    MD_agric_df = MD_agric_df.merge(weather_station_mapping_df,on = 'Field_ID', how='left')
    MD_agric_df.drop(columns="Unnamed: 0")
    MD_agric_df_weather_means = MD_agric_df.groupby("Weather_station").mean(numeric_only = True)[['Pollution_level','Rainfall', 'Ave_temps']]

    MD_agric_df_weather_means = MD_agric_df_weather_means.rename(columns = {'Ave_temps':"Temperature"})
    MD_agric_df_weather_means
<br>

## What's Next?
1. Gather all of the code from my "pipeline".

2. Re-organise the code into my three new modules: 

    a. `data_ingesation.py` - All SQL-related functions, and web-based data retrieval.

    b. `field_data_processor.py` - All transformations, cleanup, and merging functionality.

    c. `weather_data_processor.py` - All transformations and cleanup of the weather station data.

3. Copy my code into the modules and test their functionality.

4. Create automated data validation tests to ensure the data is as expected.

## Modules
    field_processor.process()
    field_df = field_processor.df

    weather_processor.process()
    field_df = field_processor.weather_df

## Data Ingestion 

1.Automating the Data Ingestion using the code that interacted with the database, and the web CSV files.

SQL:
   
    import pandas as pd # importing the Pandas package with an alias, pd
    from sqlalchemy import create_engine, text # Importing the SQL interface. If this fails, run !pip install sqlalchemy in another cell.
    import matplotlib.pyplot as plt
    import seaborn as sns


    # Create an engine for the database
    engine = create_engine('sqlite:///Maji_Ndogo_farm_survey_small.db') #Make sure to have the .db file in the same directory as this notebook, and the file name matches.


    sql_query = """
    SELECT *
    FROM geographic_features
    LEFT JOIN weather_features USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
    """

    # Create a connection object
    with engine.connect() as connection:
    
        # Use Pandas to execute the query and store the result in a DataFrame
        MD_agric_df = pd.read_sql_query(text(sql_query), connection)
<br>

CSV files: 

    weather_station_df = pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_station_data.csv")
    weather_station_mapping_df = pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv")
<br>

2.Converting the data ingestion code into functions that we can call from the module.

New Modular function:

    def create_db_engine(db_path):
        engine = create_engine(db_path)
        return engine

    def query_data(engine, sql_query):
        with engine.connect() as connection:
            df = pd.read_sql_query(text(sql_query), connection)
            return df
   <br>

So if I call:

    create_db_engine('sqlite:///Maji_Ndogo_farm_survey_small.db')

I get the SQL engine object which we can use with the query to connect to the database, and run a query.

    SQL_engine = create_db_engine('sqlite:///Maji_Ndogo_farm_survey_small.db')

    sql_query = """
    SELECT *
    FROM geographic_features
    LEFT JOIN weather_features USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
    """


    df = query_data(SQL_engine, sql_query)
    df
  <br>
  
I can even call the create_db_engine() function inside the query_data() function:

    sql_query = """
    SELECT *
    FROM geographic_features
    LEFT JOIN weather_features USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
    """


    df = query_data(create_db_engine('sqlite:///Maji_Ndogo_farm_survey_small.db'), sql_query)
    df
<br>

3.Firstly I want to add error handling into my code so that it stops the process if something is wrong, and tells me what the problem is before I continue.
Secondly, to help me understand how the code is executing I'm going to add some logs.

    import logging
    import pandas as pd

    # Name our logger so we know that logs from this module come from the data_ingestion module
    logger = logging.getLogger('data_ingestion')

    # Set a basic logging message up that prints out a timestamp, the name of our logger, and the message
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    def create_db_engine(db_path):
        try:
            engine = create_engine(db_path)
            # Test connection
            with engine.connect() as conn:
                pass
            # test if the database engine was created successfully
            logger.info("Database engine created successfully.")
            return engine # Return the engine object if it all works well
        except ImportError: #If we get an ImportError, inform the user SQLAlchemy is not installed
            logger.error("SQLAlchemy is required to use this function. Please install it first.")
            raise e
        except Exception as e:# If we fail to create an engine inform the user
            logger.error(f"Failed to create database engine. Error: {e}")
            raise e
    
    def query_data(engine, sql_query):
        try:
            with engine.connect() as connection:
                df = pd.read_sql_query(text(sql_query), connection)
            if df.empty:
                # Log a message or handle the empty DataFrame scenario as needed
                msg = "The query returned an empty DataFrame."
                logger.error(msg)
                raise ValueError(msg)
            logger.info("Query executed successfully.")
            return df
        except ValueError as e: 
            logger.error(f"SQL query failed. Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"An error occurred while querying the database. Error: {e}")
            raise e
<br>

To test it, I run an incorrect query:

    SQL_engine = create_db_engine('sqlite:///Maji_Ndogo_farm_survey_small.db')

    sql_query = """
    SELECT *
    FROM geographic_features
    LEFT JOIN weather_features USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
    WHERE Rainfall < 0 
    """
    # The last line won't ever be true, so no results will be returned. 

    df = query_data(SQL_engine, sql_query)
    df
<br>
4.I will now  get a log of what happened, and an error telling me there is something wrong with the DataFrame, and I am prevented from processing it further.

Next up, let's include the CSV data handling. This is the original code:

    weather_station_df = pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_station_data.csv")
    weather_station_mapping_df = pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv")

I'll import these two files in the same way, so I can use one function to do it.

    weather_data_URL = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_station_data.csv"
    weather_mapping_data_URL = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv"


    def read_from_web_CSV(URL):
        try:
            df = pd.read_csv(URL)
            logger.info("CSV file read successfully from the web.")
            return df
        except pd.errors.EmptyDataError as e:
            logger.error("The URL does not point to a valid CSV file. Please check the URL and try again.")
            raise e
        except Exception as e:
            logger.error(f"Failed to read CSV from the web. Error: {e}")
            raise e
    
    
    weather_df = read_from_web_CSV(weather_data_URL)
    weather_mapping_data = read_from_web_CSV(weather_mapping_data_URL)
<br>

5.Now my code can connect to a database for the field data, use a query to retrieve data and create a DataFrame. 
 I'll also import CSV files from a URL into a DataFrame, and avoid pulling unexpected data.

    from sqlalchemy import create_engine, text
    import logging
    import pandas as pd

    # Name our logger so we know that logs from this module come from the data_ingestion module
    logger = logging.getLogger('data_ingestion')
    # Set a basic logging message up that prints out a timestamp, the name of our logger, and the message
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    db_path = 'sqlite:///Maji_Ndogo_farm_survey_small.db'

    sql_query = """
    SELECT *
    FROM geographic_features
    LEFT JOIN weather_features USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
    """

    weather_data_URL = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_station_data.csv"
    weather_mapping_data_URL = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv"

    def create_db_engine(db_path):
    """

    """

        try:
            engine = create_engine(db_path)
            # Test connection
            with engine.connect() as conn:
                pass
            # test if the database engine was created successfully
            logger.info("Database engine created successfully.")
            return engine # Return the engine object if it all works well
        except ImportError: #If we get an ImportError, inform the user SQLAlchemy is not installed
            logger.error("SQLAlchemy is required to use this function. Please install it first.")
            raise e
        except Exception as e:# If we fail to create an engine inform the user
            logger.error(f"Failed to create database engine. Error: {e}")
            raise e
    
    def query_data(engine, sql_query):
        try:
            with engine.connect() as connection:
                df = pd.read_sql_query(text(sql_query), connection)
            if df.empty:
                # Log a message or handle the empty DataFrame scenario as needed
                msg = "The query returned an empty DataFrame."
                logger.error(msg)
                raise ValueError(msg)
            logger.info("Query executed successfully.")
            return df
        except ValueError as e: 
            logger.error(f"SQL query failed. Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"An error occurred while querying the database. Error: {e}")
            raise e
    
    def read_from_web_CSV(URL):
        try:
            df = pd.read_csv(URL)
            logger.info("CSV file read successfully from the web.")
            return df
        except pd.errors.EmptyDataError as e:
            logger.error("The URL does not point to a valid CSV file. Please check the URL and try again.")
            raise e
        except Exception as e:
            logger.error(f"Failed to read CSV from the web. Error: {e}")
            raise e
<br>

    # Testing module functions  
    field_df = query_data(create_db_engine(db_path), sql_query)   
    weather_df = read_from_web_CSV(weather_data_URL)
    weather_mapping_df = read_from_web_CSV(weather_mapping_data_URL)
<br>

Once we run this, I get a log telling me that it all worked.
I then run a test to make sure these functions are working well:

    field_test = field_df.shape
    weather_test = weather_df.shape
    weather_mapping_test = weather_mapping_df.shape
    print(f"field_df: {field_test}, weather_df: {weather_test}, weather_mapping_df: {weather_mapping_test}")
<br>
  field_df: (5654, 18), weather_df: (1843, 2), weather_mapping_df: (5654, 3)


6. ### Documentation
I will now create a module docstring and function docstrings for each function. 

    from sqlalchemy import create_engine, text
    import logging
    import pandas as pd

    """
    This module helps us create a db, stores the db and transforms the db.
    """

    # Name our logger so we know that logs from this module come from the data_ingestion module
    logger = logging.getLogger('data_ingestion')
    # Set a basic logging message up that prints out a timestamp, the name of our logger, and the message
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    db_path = 'sqlite:///Maji_Ndogo_farm_survey_small.db'

    sql_query = """
    SELECT *
    FROM geographic_features
    LEFT JOIN weather_features USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
    """

    weather_data_URL = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_station_data.csv"
    weather_mapping_data_URL = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv"
<br>

    squared_numbers = [i**2 for i in range(1, 11, 2)]
    print(squared_numbers)
<br>

  [1, 9, 25, 49, 81]
  

    ### START FUNCTION

    def create_db_engine(db_path):
        """
        This function creates a db from a specific file path.
        """
        try:
            engine = create_engine(db_path)
            # Test connection
            with engine.connect() as conn:
                pass
            # test if the database engine was created successfully
            logger.info("Database engine created successfully.")
            return engine # Return the engine object if it all works well
        except ImportError: #If we get an ImportError, inform the user SQLAlchemy is not installed
            logger.error("SQLAlchemy is required to use this function. Please install it first.")
            raise e
        except Exception as e:# If we fail to create an engine inform the user
            logger.error(f"Failed to create database engine. Error: {e}")
            raise e
    
    def query_data(engine, sql_query):
        """"
        This function queries the database and logs a message or handle the empty DataFrame scenario as needed.
        """
        try:
            with engine.connect() as connection:
                df = pd.read_sql_query(text(sql_query), connection)
            if df.empty:
                # Log a message or handle the empty DataFrame scenario as needed
                msg = "The query returned an empty DataFrame."
                logger.error(msg)
                raise ValueError(msg)
            logger.info("Query executed successfully.")
            return df
        except ValueError as e: 
            logger.error(f"SQL query failed. Error: {e}")
            raise e
        except Exception as e:
            logger.error(f"An error occurred while querying the database. Error: {e}")
            raise e
    
    def read_from_web_CSV(URL):
        """
        This function creates a log for reading from web content.
        """
        try:
            df = pd.read_csv(URL)
            logger.info("CSV file read successfully from the web.")
            return df
        except pd.errors.EmptyDataError as e:
            logger.error("The URL does not point to a valid CSV file. Please check the URL and try again.")
            raise e
        except Exception as e:
            logger.error(f"Failed to read CSV from the web. Error: {e}")
            raise e
    
    ### END FUNCTION
  <br>

Testing my code to make sure it works:

      # Testing module functions  
    field_df = query_data(create_db_engine(db_path), sql_query)   
    weather_df = read_from_web_CSV(weather_data_URL)
    weather_mapping_df = read_from_web_CSV(weather_mapping_data_URL)

    field_test = field_df.shape
    weather_test = weather_df.shape
    weather_mapping_test = weather_mapping_df.shape
    print(f"field_df: {field_test}, weather_df: {weather_test}, weather_mapping_df: {weather_mapping_test}")
<br>

7.Once the data_ingestion code runs smoothly, I'll create a new file, and name it data_ingestion.py and import the functions into the notebook. 
I will also have to copy over the import statements and variables.

    # Importing our new module
    from data_ingestion import create_db_engine, query_data, read_from_web_CSV

    #Checking if the function names are now associated with the module
    print(create_db_engine.__module__)
    print(query_data.__module__)
    print(read_from_web_CSV.__module__)
<br>

Now the names create_db_engine, query_data, read_from_web_CSV are linked to the data_ingestion module, so the module is imported correctly. 
So, I'll run the test commands again to make sure ite sure it works as expected

    field_df = query_data(create_db_engine(db_path), sql_query)   
    weather_df = read_from_web_CSV(weather_data_URL)
    weather_mapping_df = read_from_web_CSV(weather_mapping_data_URL)

    field_test = field_df.shape
    weather_test = weather_df.shape
    weather_mapping_test = weather_mapping_df.shape
    print(f"field_df: {field_test}, weather_df: {weather_test}, weather_mapping_df: {weather_mapping_test}")
<br>
And there we go, I have a working data ingestion module.


8. ## Field data processor
Next up, I process the field data:

    MD_agric_df = field_df.copy()

    MD_agric_df.rename(columns={'Annual_yield': 'Crop_type_Temp', 'Crop_type': 'Annual_yield'}, inplace=True)
    MD_agric_df.rename(columns={'Crop_type_Temp': 'Crop_type'}, inplace=True)
    MD_agric_df['Elevation'] = MD_agric_df['Elevation'].abs()

    # Correcting 'Crop_type' column
    def correct_crop_type(crop):
        corrections = {
            'cassaval': 'cassava',
            'wheatn': 'wheat',
            'teaa': 'tea'
        }
        return corrections.get(crop, crop)  # Get the corrected crop type, or return the original if not in corrections

    # Apply the correction function to the Crop_type column
    MD_agric_df['Crop_type'] = MD_agric_df['Crop_type'].apply(correct_crop_type)
<br>

I am going to build a Class that encapsulates the whole data processing process for the field-related data called `FieldDataProcessor`. In the class, 
I will create a DataFrame attribute and methods that alter the attribute.

    import pandas as pd
    from data_ingestion import create_db_engine, query_data, read_from_web_CSV
    import logging

    class FieldDataProcessor:

        def __init__(self, logging_level="INFO"): # When we instantiate this class, we can optionally specify what logs we want to see

            # Initialising class with attributes we need. Refer to the code above to understand how each attribute relates to the code
            self.db_path = 'sqlite:///Maji_Ndogo_farm_survey_small.db'
            self.sql_query = """
                SELECT *
                FROM geographic_features
                LEFT JOIN weather_features USING (Field_ID)
                LEFT JOIN soil_and_crop_features USING (Field_ID)
                LEFT JOIN farm_management_features USING (Field_ID)
                """
            self.columns_to_rename = {'Annual_yield': 'Crop_type', 'Crop_type': 'Annual_yield'}
            self.values_to_rename = {'cassaval': 'cassava', 'wheatn': 'wheat', 'teaa': 'tea'}
            self.weather_map_data = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv"

            self.initialize_logging(logging_level)
        
            # We create empty objects to store the DataFrame and engine in
            self.df = None
            self.engine = None
        
        # This method enables logging in the class. 
        def initialize_logging(self, logging_level):
            """
            Sets up logging for this instance of FieldDataProcessor.
            """
            logger_name = __name__ + ".FieldDataProcessor"
            self.logger = logging.getLogger(logger_name)
            self.logger.propagate = False  # Prevents log messages from being propagated to the root logger

            # Set logging level
            if logging_level.upper() == "DEBUG":
                log_level = logging.DEBUG
            elif logging_level.upper() == "INFO":
                log_level = logging.INFO
            elif logging_level.upper() == "NONE":  # Option to disable logging
                self.logger.disabled = True
                return
            else:
                log_level = logging.INFO  # Default to INFO

            self.logger.setLevel(log_level)

            # Only add handler if not already added to avoid duplicate messages
            if not self.logger.handlers:
                ch = logging.StreamHandler()  # Create console handler
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

            # Use self.logger.info(), self.logger.debug(), etc.


        # DataFrame methods 
        def ingest_sql_data(self):
            # First we want to get the data from the SQL database
            pass
    
        def rename_columns(self):
            # Annual_yield and Crop_type must be swapped
            pass
        def apply_corrections(self):
            # Correct the crop strings, Eg: 'cassaval' -> 'cassava'
            pass

        def weather_station_mapping(self):
            # Merge the weather station data to the main DataFrame
            pass

        def process(self):
            # This process calls the correct methods and applies the changes, step by step. This is the method we will call, and it will call the other methods in order
        
            weather_map_df = self.weather_station_mapping() 
            self.df = self.ingest_sql_data()
            self.df = self.rename_columns()
            self.df = self.apply_corrections()
            self.df = self.df.merge(weather_map_df, on='Field_ID', how='left')
            self.df = self.df.drop(columns="Unnamed: 0")
  <br>
  
  The idea: I will instantiate the class, and call one method, .process() to ingest and clean the data.

     # This code won't run for now, since we have not defined all of the methods.
    field_processor = FieldDataProcessor()
    field_processor.process()
<br>

.I then call the Class .df attribute, we get the DataFrame, which we can analyse.

    field_df = field_processor.df
<br>

9. ### `def ingest_sql_data()`

I'm dropping the .process() method for now and I'll add it back once it all works.

I will unscramble the code in the .ingest_sql_data() method. The method should return the initial DataFrame.

    import pandas as pd
    from data_ingestion import create_db_engine, query_data, read_from_web_CSV
    import logging

    class FieldDataProcessor:

        def __init__(self, logging_level="INFO"): # When we instantiate this class, we can optionally specify what logs we want to see

            # Initialising class with attributes we need. Refer to the code above to understand how each attribute relates to the code
            self.db_path = 'sqlite:///Maji_Ndogo_farm_survey_small.db'
            self.sql_query = """SELECT *
                FROM geographic_features
                LEFT JOIN weather_features USING (Field_ID)
                LEFT JOIN soil_and_crop_features USING (Field_ID)
                LEFT JOIN farm_management_features USING (Field_ID)
                """
            self.columns_to_rename = {'Annual_yield': 'Crop_type', 'Crop_type': 'Annual_yield'}
            self.values_to_rename = {'cassaval': 'cassava', 'wheatn': 'wheat', 'teaa': 'tea'}
            self.weather_map_data = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv"
        
            self.initialize_logging(logging_level)

            # We create empty objects to store the DataFrame and engine in
            self.df = None
            self.engine = None
        
        # This method enables logging in the class.
        def initialize_logging(self, logging_level):
            """
            Sets up logging for this instance of FieldDataProcessor.
            """
            logger_name = __name__ + ".FieldDataProcessor"
            self.logger = logging.getLogger(logger_name)
            self.logger.propagate = False  # Prevents log messages from being propagated to the root logger

            # Set logging level
            if logging_level.upper() == "DEBUG":
                log_level = logging.DEBUG
            elif logging_level.upper() == "INFO":
                log_level = logging.INFO
            elif logging_level.upper() == "NONE":  # Option to disable logging
                self.logger.disabled = True
                return
            else:
                log_level = logging.INFO  # Default to INFO

            self.logger.setLevel(log_level)

            # Only add handler if not already added to avoid duplicate messages
            if not self.logger.handlers:
                ch = logging.StreamHandler()  # Create console handler
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

            # Use self.logger.info(), self.logger.debug(), etc.


        # let's focus only on this part from now on
        def ingest_sql_data(self):
            self.engine = create_db_engine(self.db_path)
            self.df = query_data(self.engine, self.sql_query)
            self.logger.info("Sucessfully loaded data.")
            return self.df

        def rename_columns(self):
            # Annual_yield and Crop_type must be swapped
            pass
        def apply_corrections(self):
            # Correct the crop strings, Eg: 'cassaval' -> 'cassava'
            pass

        def weather_station_mapping(self):
            # Merge the weather station data to the main DataFrame
            pass

        def process(self):
            # This process calls the correct methods, and applies the changes, step by step. THis is the method we will call, and it will call the other methods in order.
            pass
<br>

Using the code to check if the code works as expected:

    field_processor = FieldDataProcessor()
    field_processor.ingest_sql_data()
    field_df = field_processor.df
    print(field_df.shape)
<br>

Output:

<Timestamp> - data_ingestion - INFO - Database engine created successfully.
<Timestamp> - data_ingestion - INFO - Query executed successfully.
<Timestamp> - __main__.FieldDataProcessor - INFO - Sucessfully loaded data.
(5654, 18)

10. ### `def rename_columns()`

Adding add rename_columns()
I'll copy the class into the top of this cell, and unscramble the code sections in the .rename_columns() method.

    class FieldDataProcessor:

        def __init__(self, logging_level="INFO"): # When we instantiate this class, we can optionally specify what logs we want to see

            # Initialising class with attributes we need. Refer to the code above to understand how each attribute relates to the code
            self.db_path = 'sqlite:///Maji_Ndogo_farm_survey_small.db'
            self.sql_query = """SELECT *
                FROM geographic_features
                LEFT JOIN weather_features USING (Field_ID)
                LEFT JOIN soil_and_crop_features USING (Field_ID)
                LEFT JOIN farm_management_features USING (Field_ID)
                """
            self.columns_to_rename = {'Annual_yield': 'Crop_type', 'Crop_type': 'Annual_yield'}
            self.values_to_rename = {'cassaval': 'cassava', 'wheatn': 'wheat', 'teaa': 'tea'}
            self.weather_map_data = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv"
        
            self.initialize_logging(logging_level)

            # We create empty objects to store the DataFrame and engine in
            self.df = None
            self.engine = None
        
        # This method enables logging in the class.
        def initialize_logging(self, logging_level):
            """
            Sets up logging for this instance of FieldDataProcessor.
            """
            logger_name = __name__ + ".FieldDataProcessor"
            self.logger = logging.getLogger(logger_name)
            self.logger.propagate = False  # Prevents log messages from being propagated to the root logger

            # Set logging level
            if logging_level.upper() == "DEBUG":
                log_level = logging.DEBUG
            elif logging_level.upper() == "INFO":
                log_level = logging.INFO
            elif logging_level.upper() == "NONE":  # Option to disable logging
                self.logger.disabled = True
                return
            else:
                log_level = logging.INFO  # Default to INFO

            self.logger.setLevel(log_level)

            # Only add handler if not already added to avoid duplicate messages
            if not self.logger.handlers:
                ch = logging.StreamHandler()  # Create console handler
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

            # Use self.logger.info(), self.logger.debug(), etc.

        # let's focus only on this part from now on
        def ingest_sql_data(self):
            self.engine = create_db_engine(self.db_path)
            self.df = query_data(self.engine, self.sql_query)
            self.logger.info("Sucessfully loaded data.")
            return self.df
    
        def rename_columns(self):

             # Extract the columns to rename from the configuration
            column1, column2 = list(self.columns_to_rename.keys())[0], list(self.columns_to_rename.values())[0]

            # Temporarily rename one of the columns to avoid a naming conflict
            temp_name = "__temp_name_for_swap__"
            while temp_name in self.df.columns:
                temp_name += "_"

                # Perform the swap
            self.df = self.df.rename(columns={column1: temp_name, column2: column1})
            self.df = self.df.rename(columns={temp_name: column2})

            self.logger.info(f"Swapped columns: {column1} with {column2}")
            
        def apply_corrections(self):
            # Correct the crop strings, Eg: 'cassaval' -> 'cassava'
            pass

        def weather_station_mapping(self):
            # Merge the weather station data to the main DataFrame
            pass

        def process(self):
            # This process calls the correct methods, and applies the changes, step by step. THis is the method we will call, and it will call the other methods in order.
            pass
<br>

Instantiate the class, connect to the database, and swap the column names:

    field_processor = FieldDataProcessor()
    field_processor.ingest_sql_data()
    field_processor.rename_columns()
    field_df = field_processor.df
    field_df['Annual_yield'].head(3)
<br>

Output:

<Timestamp> - data_ingestion - INFO - Query executed successfully.
<Timestamp> - __main__.FieldDataProcessor - INFO - Sucessfully loaded data.
<Timestamp> - __main__.FieldDataProcessor - INFO - Swapped columns: Annual_yield with Crop_type
0    0.751354
1    1.069865
2    2.208801
Name: Annual_yield, dtype: float64

11. ### `def apply_corrections()`

Here, I copy the class into the top of this cell, and fill in the <MISSING CODE> in the .apply_corrections() method.

    class FieldDataProcessor:

        def __init__(self, logging_level="INFO"): # When we instantiate this class, we can optionally specify what logs we want to see

            # Initialising class with attributes we need. Refer to the code above to understand how each attribute relates to the code
            self.db_path = 'sqlite:///Maji_Ndogo_farm_survey_small.db'
            self.sql_query = """SELECT *
                FROM geographic_features
                LEFT JOIN weather_features USING (Field_ID)
                LEFT JOIN soil_and_crop_features USING (Field_ID)
                LEFT JOIN farm_management_features USING (Field_ID)
                """
            self.columns_to_rename = {'Annual_yield': 'Crop_type', 'Crop_type': 'Annual_yield'}
            self.values_to_rename = {'cassaval': 'cassava', 'wheatn': 'wheat', 'teaa': 'tea'}
            self.weather_map_data = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv"
        
            self.initialize_logging(logging_level)

            # We create empty objects to store the DataFrame and engine in
            self.df = None
            self.engine = None
        
        # This method enables logging in the class.
        def initialize_logging(self, logging_level):
            """
            Sets up logging for this instance of FieldDataProcessor.
            """
            logger_name = __name__ + ".FieldDataProcessor"
            self.logger = logging.getLogger(logger_name)
            self.logger.propagate = False  # Prevents log messages from being propagated to the root logger

            # Set logging level
            if logging_level.upper() == "DEBUG":
                log_level = logging.DEBUG
            elif logging_level.upper() == "INFO":
                log_level = logging.INFO
            elif logging_level.upper() == "NONE":  # Option to disable logging
                self.logger.disabled = True
                return
            else:
                log_level = logging.INFO  # Default to INFO

            self.logger.setLevel(log_level)

            # Only add handler if not already added to avoid duplicate messages
            if not self.logger.handlers:
                ch = logging.StreamHandler()  # Create console handler
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

            # Use self.logger.info(), self.logger.debug(), etc.


        # let's focus only on this part from now on
        def ingest_sql_data(self):
            self.engine = create_db_engine(self.db_path)
            self.df = query_data(self.engine, self.sql_query)
            self.logger.info("Sucessfully loaded data.")
            return self.df
    
        def rename_columns(self):

             # Extract the columns to rename from the configuration
            column1, column2 = list(self.columns_to_rename.keys())[0], list(self.columns_to_rename.values())[0]
        
            # Temporarily rename one of the columns to avoid a naming conflict
            temp_name = "__temp_name_for_swap__"
            while temp_name in self.df.columns:
                temp_name += "_"

                # Extract the columns to rename from the configuration
            column1, column2 = list(self.columns_to_rename.keys())[0], list(self.columns_to_rename.values())[0]

                # Perform the swap
            self.df = self.df.rename(columns={column1: temp_name, column2: column1})
            self.df = self.df.rename(columns={temp_name: column2})

            self.logger.info(f"Swapped columns: {column1} with {column2}")
            
    
        def apply_corrections(self, column_name='Crop_type', abs_column='Elevation'):
            self.df[abs_column] = self.df[abs_column].abs()
            self.df[column_name] = self.df[column_name].apply(lambda crop: self.values_to_rename.get(crop, crop))

        def weather_station_mapping(self):
            # Merge the weather station data to the main DataFrame
            pass

        def process(self):
            # This process calls the correct methods, and applies the changes, step by step. THis is the method we will call, and it will call the other methods in order.
            pass

<br>

 Testing if the new method works:

    field_processor = FieldDataProcessor()
    field_processor.ingest_sql_data()
    field_processor.rename_columns()
    field_processor.apply_corrections()

    field_df = field_processor.df
    field_df.query("Crop_type in ['cassaval','wheatn']")
<br>

Output:
Empty DataFrame

12. ### `def weather_station_mapping()`

Now, I copy the class into the top of this cell, and fill in the <MISSING CODE> in the .weather_station_mapping() method.

    class FieldDataProcessor:

        def __init__(self, logging_level="INFO"): # When we instantiate this class, we can optionally specify what logs we want to see

            # Initialising class with attributes we need. Refer to the code above to understand how each attribute relates to the code
            self.db_path = 'sqlite:///Maji_Ndogo_farm_survey_small.db'
            self.sql_query = """SELECT *
                FROM geographic_features
                LEFT JOIN weather_features USING (Field_ID)
                LEFT JOIN soil_and_crop_features USING (Field_ID)
                LEFT JOIN farm_management_features USING (Field_ID)
                """
            self.columns_to_rename = {'Annual_yield': 'Crop_type', 'Crop_type': 'Annual_yield'}
            self.values_to_rename = {'cassaval': 'cassava', 'wheatn': 'wheat', 'teaa': 'tea'}
            self.weather_map_data = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv"
        
            self.initialize_logging(logging_level)

            # We create empty objects to store the DataFrame and engine in
            self.df = None
            self.engine = None
        
        # This method enables logging in the class.
        def initialize_logging(self, logging_level):
            """
            Sets up logging for this instance of FieldDataProcessor.
            """
            logger_name = __name__ + ".FieldDataProcessor"
            self.logger = logging.getLogger(logger_name)
            self.logger.propagate = False  # Prevents log messages from being propagated to the root logger

            # Set logging level
            if logging_level.upper() == "DEBUG":
                log_level = logging.DEBUG
            elif logging_level.upper() == "INFO":
                log_level = logging.INFO
            elif logging_level.upper() == "NONE":  # Option to disable logging
                self.logger.disabled = True
                return
            else:
                log_level = logging.INFO  # Default to INFO

            self.logger.setLevel(log_level)

            # Only add handler if not already added to avoid duplicate messages
            if not self.logger.handlers:
                ch = logging.StreamHandler()  # Create console handler
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

            # Use self.logger.info(), self.logger.debug(), etc.


        # let's focus only on this part from now on
        def ingest_sql_data(self):
            self.engine = create_db_engine(self.db_path)
            self.df = query_data(self.engine, self.sql_query)
            self.logger.info("Sucessfully loaded data.")
            return self.df
    
        def rename_columns(self):

             # Extract the columns to rename from the configuration
            column1, column2 = list(self.columns_to_rename.keys())[0], list(self.columns_to_rename.values())[0]

        
            # Temporarily rename one of the columns to avoid a naming conflict
            temp_name = "__temp_name_for_swap__"
            while temp_name in self.df.columns:
                temp_name += "_"

                # Extract the columns to rename from the configuration
            column1, column2 = list(self.columns_to_rename.keys())[0], list(self.columns_to_rename.values())[0]

                # Perform the swap
            self.df = self.df.rename(columns={column1: temp_name, column2: column1})
            self.df = self.df.rename(columns={temp_name: column2})


            self.logger.info(f"Swapped columns: {column1} with {column2}")
            
    
        def apply_corrections(self, column_name='Crop_type', abs_column='Elevation'):
            self.df[abs_column] = self.df[abs_column].abs()
            self.df[column_name] = self.df[column_name].apply(lambda crop: self.values_to_rename.get(crop, crop))

        def weather_station_mapping(self):
            weather_station_df = read_from_web_CSV(self.weather_map_data)
            self.df = pd.merge(self.df, weather_station_df[['Field_ID', 'Weather_station']], on='Field_ID', how='left')
            return read_from_web_CSV(self.weather_map_data)
    

        def process(self):
            # This process calls the correct methods and applies the changes, step by step. This is the method we will call, and it will call the other methods in order.
            pass
<br>

    field_processor = FieldDataProcessor()
    field_processor.ingest_sql_data()
    field_processor.rename_columns()
    field_processor.apply_corrections()
    field_processor.weather_station_mapping()
    field_df = field_processor.df
    field_df['Weather_station'].unique()
<br>

Output:

<Timestamp> - data_ingestion - INFO - Database engine created successfully.
<Timestamp> - data_ingestion - INFO - Query executed successfully.
<Timestamp> - __main__.FieldDataProcessor - INFO - Sucessfully loaded data.
<Timestamp> - __main__.FieldDataProcessor - INFO - Swapped columns: Annual_yield with Crop_type
<Timestamp> - data_ingestion - INFO - CSV file read successfully from the web.

array([4, 0, 1, 2, 3], dtype=int64)

13. ### `def process()`

Now I put it all together.
Copy the class into the top of this cell, and complete the .process() method.

    class FieldDataProcessor:

        def __init__(self, logging_level="INFO"): # When we instantiate this class, we can optionally specify what logs we want to see

            # Initialising class with attributes we need. Refer to the code above to understand how each attribute relates to the code
            self.db_path = 'sqlite:///Maji_Ndogo_farm_survey_small.db'
            self.sql_query = """SELECT *
                FROM geographic_features
                LEFT JOIN weather_features USING (Field_ID)
                LEFT JOIN soil_and_crop_features USING (Field_ID)
                LEFT JOIN farm_management_features USING (Field_ID)
                """
            self.columns_to_rename = {'Annual_yield': 'Crop_type', 'Crop_type': 'Annual_yield'}
            self.values_to_rename = {'cassaval': 'cassava', 'wheatn': 'wheat', 'teaa': 'tea'}
            self.weather_map_data = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv"
        
            self.initialize_logging(logging_level)

            # We create empty objects to store the DataFrame and engine in
            self.df = None
            self.engine = None
        
        # This method enables logging in the class.
        def initialize_logging(self, logging_level):
            """
            Sets up logging for this instance of FieldDataProcessor.
            """
            logger_name = __name__ + ".FieldDataProcessor"
            self.logger = logging.getLogger(logger_name)
            self.logger.propagate = False  # Prevents log messages from being propagated to the root logger

            # Set logging level
            if logging_level.upper() == "DEBUG":
                log_level = logging.DEBUG
            elif logging_level.upper() == "INFO":
                log_level = logging.INFO
            elif logging_level.upper() == "NONE":  # Option to disable logging
                self.logger.disabled = True
                return
            else:
                log_level = logging.INFO  # Default to INFO

            self.logger.setLevel(log_level)

            # Only add handler if not already added to avoid duplicate messages
            if not self.logger.handlers:
                ch = logging.StreamHandler()  # Create console handler
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

            # Use self.logger.info(), self.logger.debug(), etc.


        # let's focus only on this part from now on
        def ingest_sql_data(self):
            self.engine = create_db_engine(self.db_path)
            self.df = query_data(self.engine, self.sql_query)
            self.logger.info("Sucessfully loaded data.")
            return self.df
    
        def rename_columns(self):

             # Extract the columns to rename from the configuration
            column1, column2 = list(self.columns_to_rename.keys())[0], list(self.columns_to_rename.values())[0]

        
            # Temporarily rename one of the columns to avoid a naming conflict
            temp_name = "__temp_name_for_swap__"
            while temp_name in self.df.columns:
                temp_name += "_"

                # Extract the columns to rename from the configuration
            column1, column2 = list(self.columns_to_rename.keys())[0], list(self.columns_to_rename.values())[0]

                # Perform the swap
            self.df = self.df.rename(columns={column1: temp_name, column2: column1})
            self.df = self.df.rename(columns={temp_name: column2})


            self.logger.info(f"Swapped columns: {column1} with {column2}")
            
    
        def apply_corrections(self, column_name='Crop_type', abs_column='Elevation'):
            self.df[abs_column] = self.df[abs_column].abs()
            self.df[column_name] = self.df[column_name].apply(lambda crop: self.values_to_rename.get(crop, crop))

        def weather_station_mapping(self):
            weather_station_df = read_from_web_CSV(self.weather_map_data)
            self.df = pd.merge(self.df,weather_station_df[['Field_ID','Weather_station']], on='Field_ID', how='left')
            return read_from_web_CSV(self.weather_map_data)
    
        def process(self):
            self.ingest_sql_data()
            field_processor.rename_columns()
            field_processor.apply_corrections()
            field_processor.weather_station_mapping()
<br>

    field_processor = FieldDataProcessor()
    field_processor.process()

    field_df = field_processor.df
    field_df['Weather_station'].unique()
<br>

Output:

<Timestamp>  - data_ingestion - INFO - Database engine created successfully.
<Timestamp>  - data_ingestion - INFO - Query executed successfully.
<Timestamp>  - __main__.FieldDataProcessor - INFO - Sucessfully loaded data.
<Timestamp>  - __main__.FieldDataProcessor - INFO - Swapped columns: Annual_yield with Crop_type
<Timestamp> - data_ingestion - INFO - CSV file read successfully from the web.

array([4, 0, 1, 2, 3], dtype=int64)

14. ### Centralising the data pipeline configuration details

Add the configuration details from the data_ingestion.py module into the config_params dictionary.

    config_params = {
        "sql_query": """
    SELECT *
    FROM geographic_features
    LEFT JOIN weather_features USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
                """, # Insert your SQL query
        "db_path": 'sqlite:///Maji_Ndogo_farm_survey_small.db', # Insert the db_path of the database
        "columns_to_rename": {'Annual_yield': 'Crop_type', 'Crop_type': 'Annual_yield'}, # Insert the disctionary of columns we want to swop the names of 
        "values_to_rename": {'cassaval': 'cassava', 'wheatn': 'wheat', 'teaa': 'tea'}, # Insert the croptype renaming dictionary
        "weather_csv_path": "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_station_data.csv", # Insert the weather data CSV here
        "weather_mapping_csv": "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv", # Insert the weather data mapping CSV here
    }
<br>

15.Removing lines form the data_ingestion.py module file, since I am calling them from the FieldDataProcessor class.

    # Remove these lines from the data_ingestion.py module
    db_path = 'sqlite:///Maji_Ndogo_farm_survey_small.db'

    sql_query = """
    SELECT *
    FROM geographic_features
    LEFT JOIN weather_features USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
    """

    weather_data_URL = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_station_data.csv"
    weather_mapping_data_URL = "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv"
<br>

16.Next, I alter the attributes of the FieldDataProcessor class to reference the config_params dictionary instead. And then add config_params as a parameter to the class instantiation method.

    ### START FUNCTION
    class FieldDataProcessor:

        def __init__(self, config_params, logging_level="INFO"):  # Make sure to add this line, passing in config_params to the class 
            self.db_path = config_params["db_path"]
            self.sql_query = config_params["sql_query"]
            self.columns_to_rename = config_params["columns_to_rename"]
            self.values_to_rename = config_params["values_to_rename"]
            self.weather_map_data = config_params["weather_mapping_csv"]
            self.weather_data = config_params["weather_csv_path"]

            # Add the rest of your class code here

            self.initialize_logging(logging_level)

            # We create empty objects to store the DataFrame and engine in
            self.df = None
            self.engine = None
        
        # This method enables logging in the class.
        def initialize_logging(self, logging_level):
            """
            Sets up logging for this instance of FieldDataProcessor.
            """
            logger_name = __name__ + ".FieldDataProcessor"
            self.logger = logging.getLogger(logger_name)
            self.logger.propagate = False  # Prevents log messages from being propagated to the root logger

            # Set logging level
            if logging_level.upper() == "DEBUG":
                log_level = logging.DEBUG
            elif logging_level.upper() == "INFO":
                log_level = logging.INFO
            elif logging_level.upper() == "NONE":  # Option to disable logging
                self.logger.disabled = True
                return
            else:
                log_level = logging.INFO  # Default to INFO

            self.logger.setLevel(log_level)

            # Only add handler if not already added to avoid duplicate messages
            if not self.logger.handlers:
                ch = logging.StreamHandler()  # Create console handler
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

            # Use self.logger.info(), self.logger.debug(), etc.


        # let's focus only on this part from now on
        def ingest_sql_data(self):
            self.engine = create_db_engine(self.db_path)
            self.df = query_data(self.engine, self.sql_query)
            self.logger.info("Sucessfully loaded data.")
            return self.df
    
        def rename_columns(self):

             # Extract the columns to rename from the configuration
            column1, column2 = list(self.columns_to_rename.keys())[0], list(self.columns_to_rename.values())[0]

        
            # Temporarily rename one of the columns to avoid a naming conflict
            temp_name = "__temp_name_for_swap__"
            while temp_name in self.df.columns:
                temp_name += "_"

                # Extract the columns to rename from the configuration
            column1, column2 = list(self.columns_to_rename.keys())[0], list(self.columns_to_rename.values())[0]

                # Perform the swap
            self.df = self.df.rename(columns={column1: temp_name, column2: column1})
            self.df = self.df.rename(columns={temp_name: column2})


            self.logger.info(f"Swapped columns: {column1} with {column2}")
            
    
        def apply_corrections(self, column_name='Crop_type', abs_column='Elevation'):
            self.df[abs_column] = self.df[abs_column].abs()
            self.df[column_name] = self.df[column_name].apply(lambda crop: self.values_to_rename.get(crop, crop))

        def weather_station_mapping(self):
            weather_station_df = read_from_web_CSV(self.weather_map_data)
            self.df = pd.merge(self.df,weather_station_df[['Field_ID','Weather_station']], on='Field_ID', how='left')
            return read_from_web_CSV(self.weather_map_data)
    
        def process(self):
            self.ingest_sql_data()
            #Insert your code here
            field_processor.rename_columns()
            field_processor.apply_corrections()
            field_processor.weather_station_mapping()
      
        
    ### END FUNCTION
<br>

17.Instantiating the class with the new dictionary

    config_params = {
        "sql_query": """
    SELECT *
    FROM geographic_features
    LEFT JOIN weather_features USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
                """, # Insert your SQL query
        "db_path": 'sqlite:///Maji_Ndogo_farm_survey_small.db', # Insert the db_path of the database
        "columns_to_rename": {'Annual_yield': 'Crop_type', 'Crop_type': 'Annual_yield'}, # Insert the disctionary of columns we want to swop the names of 
        "values_to_rename": {'cassaval': 'cassava', 'wheatn': 'wheat', 'teaa': 'tea'}, # Insert the croptype renaming dictionary
        "weather_csv_path": "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_station_data.csv", # Insert the weather data CSV here
        "weather_mapping_csv": "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv", # Insert the weather data mapping CSV here
    }

    field_processor = FieldDataProcessor(config_params)
    field_processor.process()

    field_df = field_processor.df
    field_df['Weather_station'].unique()
<br>

Output:

<Timestamp>  - data_ingestion - INFO - Database engine created successfully.
<Timestamp>  - data_ingestion - INFO - Query executed successfully.
<Timestamp>  - __main__.FieldDataProcessor - INFO - Sucessfully loaded data.
<Timestamp>  - __main__.FieldDataProcessor - INFO - Swapped columns: Annual_yield with Crop_type
<Timestamp> - data_ingestion - INFO - CSV file read successfully from the web.

array([4, 0, 1, 2, 3], dtype=int64)

