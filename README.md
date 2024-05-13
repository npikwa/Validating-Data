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

1.  ### Automating the Data Ingestion using the code that interacted with the database, and the web CSV files.

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

2.  ### Converting the data ingestion code into functions that we can call from the module.

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

3.   ### Firstly I want to add error handling into my code so that it stops the process if something is wrong, and tells me what the problem is before I continue.
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

4.  ### I will now  get a log of what happened, and an error telling me there is something wrong with the DataFrame, and I am prevented from processing it further.

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

5.   ### Now my code can connect to a database for the field data, use a query to retrieve data and create a DataFrame. 
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


6.  ### Documentation
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

7.   ### Once the data_ingestion code runs smoothly, I'll create a new file, and name it data_ingestion.py and import the functions into the notebook. 
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


8.  ## Field data processor
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

9.  ### `def ingest_sql_data()`

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

10.  ### `def rename_columns()`

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

11.  ### `def apply_corrections()`

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

12.  ### `def weather_station_mapping()`

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

13.  ### `def process()`

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

14.  ### Centralising the data pipeline configuration details

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

15.  ### Removing lines form the data_ingestion.py module file, since I am calling them from the FieldDataProcessor class.

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

16.  ### Next, I alter the attributes of the FieldDataProcessor class to reference the config_params dictionary instead. And then add config_params as a parameter to the class instantiation method.

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

17.  ### Instantiating the class with the new dictionary

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

18.  ### Creating `field_data_processor.py`

Including all of the required content, ensuring the module is PEP 8 complient, including all imports and parameter definitions, 
and creating the `field_data_processor.py` module file.

    import re # Importing all the packages we will use eventually
    import numpy as np
    import pandas as pd
    from field_data_processor import FieldDataProcessor # Importing our new module
    import logging 

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config_params =  {
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
    }# Paste in your config_params dictionary here


    # Instantiating the class with config_params passed to the class as a parameter 
    field_processor = FieldDataProcessor(config_params)
    field_processor.process()
    field_df = field_processor.df

    # Test
    field_df['Weather_station'].unique()
<br>

Output:

<Timestamp>  - data_ingestion - INFO - Database engine created successfully.
<Timestamp>  - data_ingestion - INFO - Query executed successfully.
<Timestamp>  - field_data_processor.FieldDataProcessor - INFO - Sucessfully loaded data.
<Timestamp>  - field_data_processor.FieldDataProcessor - INFO - Swapped columns: Annual_yield with Crop_type
<Timestamp> - data_ingestion - INFO - CSV file read successfully from the web.

array([4, 0, 1, 2, 3], dtype=int64)

19.  ## Weather data processor

Now for the last module. The WeatherDataProcessor class will be dealing with all of the weather-related data. Again I need to instantiate the class, then call a .process() method to import and clean the data. 

    import re # Importing the regex pattern
    import numpy as np

    weather_station_df = pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_station_data.csv")
    weather_station_mapping_df = pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv")

    patterns = {
        'Rainfall': r'(\d+(\.\d+)?)\s?mm',
         'Temperature': r'(\d+(\.\d+)?)\s?C',
        'Pollution_level': r'=\s*(-?\d+(\.\d+)?)|Pollution at \s*(-?\d+(\.\d+)?)'
         }

    def extract_measurement(message):
        """
        Extracts a numeric measurement value from a given message string.

        The function applies regular expressions to identify and extract
        numeric values related to different types of measurements such as
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

    # The function creates a tuple with the measurement type and value into a Pandas Series
    result = weather_station_df['Message'].apply(extract_measurement)

    # Create separate columns for 'Measurement' and 'extracted_value' by unpacking the tuple with Lambda functions.
    weather_station_df['Measurement'] = result.apply(lambda x: x[0])
    weather_station_df['Value'] = result.apply(lambda x: x[1])
<br>

20.  ### Completing the values for the new keys in `config_params`.

Here's the completed config_params dictionary:

    weather_station_df = pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_station_data.csv")
    weather_station_mapping_df = pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv")

    patterns = {
        'Rainfall': r'(\d+(\.\d+)?)\s?mm',
        'Temperature': r'(\d+(\.\d+)?)\s?C',
        'Pollution_level': r'=\s*(-?\d+(\.\d+)?)|Pollution at \s*(-?\d+(\.\d+)?)'
    }

    config_params = {
        "sql_query": """
    SELECT *
    FROM geographic_features
    LEFT JOIN weather_features USING (Field_ID)
    LEFT JOIN soil_and_crop_features USING (Field_ID)
    LEFT JOIN farm_management_features USING (Field_ID)
               """,
        "db_path": 'sqlite:///Maji_Ndogo_farm_survey_small.db',
        "columns_to_rename": {'Annual_yield': 'Crop_type', 'Crop_type': 'Annual_yield'},
        "values_to_rename": {'cassaval': 'cassava', 'wheatn': 'wheat', 'teaa': 'tea'},
        "weather_csv_path": "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_station_data.csv",
        "weather_mapping_csv": "https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Maji_Ndogo/Weather_data_field_mapping.csv",
        "regex_patterns": {
            'Rainfall': r'(\d+(\.\d+)?)\s?mm',
            'Temperature': r'(\d+(\.\d+)?)\s?C',
            'Pollution_level': r'=\s*(-?\d+(\.\d+)?)|Pollution at \s*(-?\d+(\.\d+)?)'
        }
<br>

21.  ### Completing the `weather_data_processor` module. Including all of the required content, ensuring the module is PEP 8 compliant, including all imports and parameter definitions, and creating the `weather_data_processor.py` module file.

    # These are the imports we're going to use in the weather data processing module
    import re
    import numpy as np
    import pandas as pd
    import logging
    from data_ingestion import read_from_web_CSV
<br>

    ### START FUNCTION 

    class WeatherDataProcessor:
        def __init__(self, config_params, logging_level="INFO"): # Now we're passing in the confi_params dictionary already
            self.weather_station_data = config_params['weather_csv_path']
            self.patterns = config_params['regex_patterns']
            self.weather_df = None  # Initialize weather_df as None or as an empty DataFrame
            self.initialize_logging(logging_level)

        def initialize_logging(self, logging_level):
            logger_name = __name__ + ".WeatherDataProcessor"
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

        def weather_station_mapping(self):
            self.weather_df = read_from_web_CSV(self.weather_station_data)
            self.logger.info("Successfully loaded weather station data from the web.") 
            # Here, you can apply any initial transformations to self.weather_df if necessary.

    
        def extract_measurement(self, message):
            for key, pattern in self.patterns.items():
                match = re.search(pattern, message)
                if match:
                    self.logger.debug(f"Measurement extracted: {key}")
                    return key, float(next((x for x in match.groups() if x is not None)))
            self.logger.debug("No measurement match found.")
            return None, None

        def process_messages(self):
            if self.weather_df is not None:
                result = self.weather_df['Message'].apply(self.extract_measurement)
                self.weather_df['Measurement'], self.weather_df['Value'] = zip(*result)
                self.logger.info("Messages processed and measurements extracted.")
            else:
                self.logger.warning("weather_df is not initialized, skipping message processing.")
            return self.weather_df

        def calculate_means(self):
            if self.weather_df is not None:
                means = self.weather_df.groupby(by=['Weather_station_ID', 'Measurement'])['Value'].mean()
                self.logger.info("Mean values calculated.")
                return means.unstack()
            else:
                self.logger.warning("weather_df is not initialized, cannot calculate means.")
                return None
    
        def process(self):
            self.weather_station_mapping()  # Load and assign data to weather_df
            self.process_messages()  # Process messages to extract measurements
            self.logger.info("Data processing completed.")
    ### END FUNCTION
<br>

 22.  ### Running a code to import the new module, and make sure the module worked correctly.

    import re
    import numpy as np
    import pandas as pd
    # from field_data_processor import FieldDataProcessor
    from weather_data_processor import WeatherDataProcessor
    import logging 

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config_params = # Paste in your config_params dictionary here

    # Ignoring the field data for now.
    # field_processor = FieldDataProcessor(config_params)
    # field_processor.process()
    # field_df = field_processor.df

    weather_processor = WeatherDataProcessor(config_params)
    weather_processor.process()
    weather_df = weather_processor.weather_df

    weather_df['Measurement'].unique()
<br>

Output:

<Timestamp> - data_ingestion - INFO - CSV file read successfully from the web.
<Timestamp> - __main__.WeatherDataProcessor  - INFO - Successfully loaded weather station data from the web.
<Timestamp> - __main__.WeatherDataProcessor  - INFO - Messages processed and measurements extracted.
<Timestamp> - __main__.WeatherDataProcessor  - INFO - Data processing completed.

array(['Temperature', 'Pollution_level', 'Rainfall'], dtype=object)

23.  ### Validating the data pipeline
Get the data:

    import re
    import numpy as np
    import pandas as pd
    from field_data_processor import FieldDataProcessor
    from weather_data_processor import WeatherDataProcessor
    import logging 

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config_params = # Paste in your config_params dictionary here

    field_processor = FieldDataProcessor(config_params)
    field_processor.process()
    field_df = field_processor.df

    weather_processor = WeatherDataProcessor(config_params)
    weather_processor.process()
    weather_df = weather_processor.weather_df
<br>

24.  ### Running the next code to create CSV files, run `pytest` in the terminal using `!pytest validate_data.py -v`, and delete the CSV files once the test is complete.

    # !pip install pytest

    weather_df.to_csv('sampled_weather_df.csv', index=False)
    field_df.to_csv('sampled_field_df.csv', index=False)

    !pytest validate_data.py -v

    import os# Define the file paths
    weather_csv_path = 'sampled_weather_df.csv'
    field_csv_path = 'sampled_field_df.csv'

    # Delete sampled_weather_df.csv if it exists
    if os.path.exists(weather_csv_path):
        os.remove(weather_csv_path)
        print(f"Deleted {weather_csv_path}")
    else:
        print(f"{weather_csv_path} does not exist.")

    # Delete sampled_field_df.csv if it exists
    if os.path.exists(field_csv_path):
        os.remove(field_csv_path)
        print(f"Deleted {field_csv_path}")
    else:
        print(f"{field_csv_path} does not exist.")
<br>

Output:

============================ test session starts =============================
platform win32 -- Python 3.12.1, pytest-8.0.0, pluggy-1.4.0 -- ...
cachedir: .pytest_cache
rootdir: ...
plugins: anyio-4.2.0
collecting ... collected 7 items

validate_data.py::test_read_weather_DataFrame_shape PASSED               [ 14%]
validate_data.py::test_read_field_DataFrame_shape PASSED                 [ 28%]
validate_data.py::test_weather_DataFrame_columns PASSED                  [ 42%]
validate_data.py::test_field_DataFrame_columns PASSED                    [ 57%]
validate_data.py::test_field_DataFrame_non_negative_elevation PASSED     [ 71%]
validate_data.py::test_crop_types_are_valid PASSED                       [ 85%]
validate_data.py::test_positive_rainfall_values PASSED                   [100%]

============================== warnings summary ===============================
..\..\..\..\..\..\..\..\anaconda3\envs\Latest\Lib\site-packages\dateutil\tz\tz.py:37
  ...: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
    EPOCH = datetime.datetime.utcfromtimestamp(0)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 7 passed, 1 warning in 1.13s =========================
Deleted sampled_weather_df.csv
Deleted sampled_field_df.csv
⚠️ Depending on the version of Python, there may be various warnings like the one above. These are normally DeprecationWarnings so we can safely ignore these for now. We're interested in whether all the dataset tests passed.

25.  ### Validating the dataset

    import re
    import numpy as np
    import pandas as pd
    from field_data_processor import FieldDataProcessor
    from weather_data_processor import WeatherDataProcessor
    import logging 

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config_params =    # Paste in your previous dictionary data in here

    field_processor = FieldDataProcessor(config_params)
    field_processor.process()
    field_df = field_processor.df

    weather_processor = WeatherDataProcessor(config_params)
    weather_processor.process()
    weather_df = weather_processor.weather_df

    # Rename 'Ave_temps' in field_df to 'Temperature' to match weather_df
    field_df.rename(columns={'Ave_temps': 'Temperature'}, inplace=True)
<br>

## Hypothesis

The null hypothesis $H_0$ is testing if the field data is representing the reality in Maji Ndogo by looking at an independent set of data. If the field data (means) are the same as the weather data (means), then it indicates no significant difference between the datasets. Essentially,  any difference we see between these means is because of randomness. However, if the means differ significantly, we'll know there is a reason for it, and that it is not just a random fluctuation in the data.

Given a significance level $\alpha$ of 0.05 for a two-tailed test, we have the following conditions for our hypothesis test at a 95% confidence interval:
- $H_0$: There is no significant difference between the means of the two datasets. This is expressed as $\mu_{field} = \mu_{weather}$.
- $H_a$: There is a significant difference between the means of the two datasets. This is expressed as $\mu_{field} \neq \mu_{weather}$.

If the p-value obtained from the test:
- is less than or equal to the significance level, so $p \leq \alpha$, we reject the null hypothesis.
- is larger than the significance level, so $p > \alpha$, we cannot reject the null hypothesis, as we cannot find a statistically significant difference between the datasets at the 95% confidence level.

<br>

### First, I'm going to import all of the packages and define a few variables
Importing a new method, .ttest_ind(). This method takes in two data columns and calculates means, variance, and returns the the t- and p-statistics.
I will also use the two-sided t-test, by adding  the `alternative = 'two-sided'` keyword.

    from scipy.stats import ttest_ind
    import numpy as np

    # Now, the measurements_to_compare can directly use 'Temperature', 'Rainfall', and 'Pollution_level'
    measurements_to_compare = ['Temperature', 'Rainfall', 'Pollution_level']
<br>

## Creating a filter_field_data function that takes in the field_df DataFrame, the station_id, and measurement type, and retuns a single column (series) of data filtered by the station_id, and measurement.

    ### START FUNCTION
    def filter_field_data(df, station_id, measurement):
        """
        Filter field data by station_id and measurement type.

        Args:
        - df (DataFrame): DataFrame containing field data.
        - station_id (str): Station ID to filter by.
        - measurement (str): Measurement type to filter by.

        Returns:
        - filtered_data (Series): Single column of data filtered by station_id and measurement type.
        """
        filtered_data = df[(df['Station_ID'] == station_id) & (df['Measurement'] == measurement)]['Value']
        return filtered_data

    ### END FUNCTION
<br>

### Example 1:

    # Example for station ID 0 and Temperature
    station_id = 0
    alpha = 0.05
    measurement = 'Temperature'

    # Filter data for the specific station and measurement
    field_values = filter_field_data(field_df, station_id, measurement)
    field_values

Output:

1       13.35
2       13.30
8       12.80
10      13.70
14      13.35
        ...  
5627    13.30
5630    14.25
5632    11.00
5638    13.30
5642    12.85
Name: Temperature, Length: 1375, dtype: float64

### Example 2

    station_id = 0
    alpha = 0.05
    measurement = 'Temperature'

    # Filter data for the specific station and measurement
    field_values = filter_field_data(field_df, station_id, measurement)
    print(f"Shape: {field_values.shape}, First value: {field_values.iloc[0]} ")

Output:

Shape: (1375,), First value: 13.35

### Creating a data filter function that takes in the `weather_df` DataFrame, the `station_id`, and `measurement` type, and returns a **single column** (series) of data filtered by the `station_id`, and `measurement`.

    ### START FUNCTION
    def filter_weather_data(df, station_id, measurement):
        """
        Filter weather data by station_id and measurement type.

        Args:
        - df (DataFrame): DataFrame containing weather data.
        - station_id (str): Station ID to filter by.
        - measurement (str): Measurement type to filter by.

        Returns:
        - filtered_data (Series): Single column of data filtered by station_id and measurement type.
        """
        filtered_data = df[(df['Station_ID'] == station_id) & (df['Measurement'] == measurement)]['Value']
        return filtered_data

    ### END FUNCTION
<br>

 ### Running a code to import the new module, and make sure our module worked correctly.

### Example 1:

     # Example for station ID 0 and Temperature
    station_id = 0
    alpha = 0.05
    measurement = 'Temperature'

    # Filter data for the specific station and measurement

    weather_values = filter_weather_data(weather_df, station_id, measurement)
    weather_values

Output:

0       12.82
2       14.53
29      14.28
32      12.87
67      13.13
        ...  
1804    12.77
1805    14.13
1817    13.14
1833    14.14
1834    13.61
Name: Value, Length: 100, dtype: float64

### Example 2:

    # Example for station ID 0 and Temperature
    station_id = 0
    alpha = 0.05
    measurement = 'Temperature'

    # Filter data for the specific station and measurement

    weather_values = filter_weather_data(weather_df, station_id, measurement)

    print(f"Shape: {weather_values.shape}, First value: {weather_values.iloc[0]}")

 Output: 

 Shape: (100,), First value: 12.82 

### Creating a function that calculates the t-statistic and p-value. The function should accept two single columns of data and return a tuple of the t-statistic and p-value.

    ### START FUNCTION
    from scipy.stats import ttest_ind

    def run_ttest(Column_A, Column_B):
        """
        Calculate the t-statistic and p-value for two samples.

        Args:
        - Column_A (Series): First sample data.
        - Column_B (Series): Second sample data.

        Returns:
        - t_statistic (float): The calculated t-statistic.
        - p_value (float): The calculated p-value.
        """
        t_statistic, p_value = ttest_ind(Column_A, Column_B)
        return t_statistic, p_value

    ### END FUNCTION

### Example:

    # Example for station ID 0 and Temperature
    station_id = 0
    alpha = 0.05
    measurement = 'Temperature'

    # Filter data for the specific station and measurement
    field_values = filter_field_data(field_df, station_id, measurement)
    weather_values = filter_weather_data(weather_df, station_id, measurement)

    # Perform t-test
    t_stat, p_val = run_ttest(field_values, weather_values)
    print(f"T-stat: {t_stat:.5f}, p-value: {p_val:.5f}")

Output:

T-stat: -0.11632, p-value: 0.90761

### The t-test 

    ### START FUNCTION

    def print_ttest_results(station_id, measurement, p_val, alpha):
        """
        Interprets and prints the results of a t-test based on the p-value.
        """
        if p_val < alpha:
            print(f"   Significant difference in {measurement} detected at Station {station_id}, (P-Value: {p_val:.5f} < {alpha}). Null hypothesis rejected.")
        else:
            print(f"   No significant difference in {measurement} detected at Station {station_id}, (P-Value: {p_val:.5f} > {alpha}). Null hypothesis not rejected.")

    ### END FUNCTION

  ### Example: 

    # Example for station ID 0 and Temperature
    station_id = 0

    measurement = 'Temperature'

    # Filter data for the specific station and measurement
    field_values = filter_field_data(field_df, station_id, measurement)
    weather_values = filter_weather_data(weather_df, station_id, measurement)

    # Perform t-test
    t_stat, p_val = run_ttest(field_values, weather_values)
    print_ttest_results(station_id, measurement, p_val, alpha)

Output:

No significant difference in Temperature detected (P-Value: 0.90761 > 0.05). Null hypothesis not rejected.

### Creating a function that loops over measurements_to_compare and all station_id, perform a t-test and print the results. The function should accept field_df, weather_df, list_measurements_to_compare, alpha. the value of alpha should default to a value of 0.05. Hint: use print_ttest_results().

    ### START FUNCTION
    def hypothesis_results(field_df, weather_df, list_measurements_to_compare, alpha=0.05):
        """
        Perform t-tests for each measurement in list_measurements_to_compare between field_df and weather_df
        and print the results.

        Args:
        - field_df (DataFrame): DataFrame containing field data.
        - weather_df (DataFrame): DataFrame containing weather data.
        - list_measurements_to_compare (list): List of measurements to compare.
        - alpha (float, optional): Significance level for t-tests. Default is 0.05.

        Returns:
        - None
        """
        for measurement in list_measurements_to_compare:
            print(f"Comparing measurement: {measurement}")
            for station_id in field_df['Station_ID'].unique():
                field_data = field_df[(field_df['Station_ID'] == station_id) & (field_df['Measurement'] == measurement)]['Value']
                weather_data = weather_df[(weather_df['Station_ID'] == station_id) & (weather_df['Measurement'] == measurement)]['Value']
                if len(field_data) > 1 and len(weather_data) > 1:
                    t_statistic, p_value = run_ttest(field_data, weather_data)
                    print_ttest_results(station_id, measurement, p_value, alpha)
                else:
                    print(f"   Insufficient data for comparison at Station {station_id} for {measurement}.")

    ### END FUNCTION

### Example:

    alpha = 0.05
    hypothesis_results(field_df, weather_df, measurements_to_compare, alpha)

Output:

 No significant difference in Temperature detected at Station 0, (P-Value: 0.90761 > 0.05). Null hypothesis not rejected.
 <br>
 No significant difference in Rainfall detected at Station 0, (P-Value: 0.21621 > 0.05). Null hypothesis not rejected.
 <br>
 No significant difference in Pollution_level detected at Station 0, (P-Value: 0.56418 > 0.05). Null hypothesis not rejected.
 <br>
 No significant difference in Temperature detected at Station 1, (P-Value: 0.47241 > 0.05). Null hypothesis not rejected.
 <br>
No significant difference in Rainfall detected at Station 1, (P-Value: 0.54499 > 0.05). Null hypothesis not rejected.
 <br>
No significant difference in Pollution_level detected at Station 1, (P-Value: 0.24410 > 0.05). Null hypothesis not rejected.
 <br>
No significant difference in Temperature detected at Station 2, (P-Value: 0.88671 > 0.05). Null hypothesis not rejected.
 <br>
No significant difference in Rainfall detected at Station 2, (P-Value: 0.36466 > 0.05). Null hypothesis not rejected.
 <br>
No significant difference in Pollution_level detected at Station 2, (P-Value: 0.99388 > 0.05). Null hypothesis not rejected.
 <br>
No significant difference in Temperature detected at Station 3, (P-Value: 0.66445 > 0.05). Null hypothesis not rejected.
<br>
No significant difference in Rainfall detected at Station 3, (P-Value: 0.39847 > 0.05). Null hypothesis not rejected.
<br>
No significant difference in Pollution_level detected at Station 3, (P-Value: 0.15466 > 0.05). Null hypothesis not rejected.
 <br>
No significant difference in Temperature detected at Station 4, (P-Value: 0.88575 > 0.05). Null hypothesis not rejected.
 <br>
No significant difference in Rainfall detected at Station 4, (P-Value: 0.33237 > 0.05). Null hypothesis not rejected.
 <br>
No significant difference in Pollution_level detected at Station 4, (P-Value: 0.21508 > 0.05). Null hypothesis not rejected.

## Conclusion

### For all of our measurements the p-value > alpha, so there is not enough evidence to reject the null hypothesis. This means we have no evidence to suggest that the weather data is different from the field data. This makes us confident that our field data, at least in terms of temperature, rainfall, and pollution level is reflecting the reality. From the EDA showed that there were some relationships, and possible correlations with the standard yield, but we really can't say what affects a crop's success, because all of them seemed to. In a sense, we could not clearly see the relationships, if we were given a set of conditions like rainfall, pH, and crop type, we could not reliably estimate what the standard yield of a crop is, because the relationships are hard to understand. So, we allow a machine to look for patterns, because computers are not limited to three dimensions, they can calculate for hours, and find hidden patterns we cannot.
