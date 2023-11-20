# Import convention
import warnings
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

def main():
    # Start of program
    st.title('Milestone #2')
    modified_data_agn = None
    uploaded_data = upload_dataset()
    st.divider()

    # EDA
    if not uploaded_data is None:
        st.header('Exploratory Data Analysis:')
        modified_data = feature_engineering(uploaded_data)
        analyze_data(modified_data, uploaded_data)
        modified_data_agn = categorize_obesity(modified_data)
        st.subheader('Encode Data')
        x_train, x_test, y_train, y_test = split_data(modified_data_agn)
        x_te = []
        x_te_patientID = x_test[:, 0]
        x_te = pd.DataFrame({'Patient ID': x_te_patientID})
        x_train1 = encode_data(pd.DataFrame(x_train))
        x_test1 = encode_data(pd.DataFrame(x_test))


    if not modified_data_agn is None:
        st.header('Machine Learning:')
        selected_tab = st.selectbox("Select Model", ["Decision Tree Model"])
        if selected_tab == "Decision Tree Model":
            decision_tree(x_train1, y_train, x_test1, y_test, x_te)

def upload_dataset():
    uploaded_file = st.file_uploader('Upload Milestone #2 Dataset',['csv', 'txt'], 
                    help='Single-file upload only')
    if uploaded_file is not None:
        # Check if the file type is supported
        if uploaded_file.type in ['application/vnd.ms-excel', 'text/csv']:
            data = pd.read_csv(uploaded_file)

            # Display the raw data
            st.write(f"**Raw DataFrame - {uploaded_file.name}**")
            st.write(data)

            # List out columns with missing values
            missing_values = data.isnull().sum()

            # Check if there are any missing values
            if missing_values.sum() == 0:
                st.write("No missing values found.")
            else:
                for column, count in missing_values.items():
                    if count > 0:
                        with st.expander("Columns with missing values:"):
                            st.write(f" - [{column}] has {count} missing values.")

            # Returns the DataFrame if the file is successfully loaded
            return data
        else:
            st.warning(f"Unsupported file type: {uploaded_file.type}")


def feature_engineering(uploaded_data):
    columns_to_remove = st.multiselect("Select columns to remove", list(uploaded_data.columns))
    if columns_to_remove:
        # Remove selected columns from the DataFrame
        modified_data = uploaded_data.drop(columns=columns_to_remove, axis=1)
    else:
        modified_data = uploaded_data

    if 'Age' in modified_data.columns:
        # Round the age column values to the nearest whole number and convert to int
        modified_data['Age'] = round(modified_data['Age']).astype(int)

        for dataset in [modified_data]:
            dataset.loc[(dataset['Age'] >= 14) & (dataset['Age'] <= 21), 'Age'] = 0
            dataset.loc[(dataset['Age'] > 21) & (dataset['Age'] <= 55), 'Age'] = 1
            dataset.loc[dataset['Age'] > 55, 'Age'] = 2

            dataset['Age'] = dataset['Age'].astype(str)
            dataset.loc[dataset['Age'] == '0', 'Age'] = "Teens"
            dataset.loc[dataset['Age'] == '1', 'Age'] = "Adults"
            dataset.loc[dataset['Age'] == '2', 'Age'] = "Elderly"

    if 'Height' in modified_data.columns:
        for dataset in [modified_data]:
            dataset.loc[(dataset['Height'] >= 1.4) & (dataset['Height'] <= 1.6), 'Height'] = 0
            dataset.loc[(dataset['Height'] > 1.6) & (dataset['Height'] <= 1.8), 'Height'] = 1
            dataset.loc[dataset['Height'] > 1.8, 'Height'] = 2

            dataset['Height'].astype(str)
            dataset.loc[ dataset['Height'] == 0, 'Height'] = "low"
            dataset.loc[ dataset['Height'] == 1, 'Height'] = "mid"
            dataset.loc[ dataset['Height'] == 2, 'Height'] = "high"

    if 'Weight' in modified_data.columns:
        for dataset in [modified_data]:
            dataset.loc[(dataset['Weight'] >= 30) & (dataset['Weight'] <= 70), 'Weight'] = 0
            dataset.loc[(dataset['Weight'] > 70) & (dataset['Weight'] <= 110), 'Weight'] = 1
            dataset.loc[dataset['Weight'] > 110, 'Weight'] = 2

        dataset['Weight'].astype(str)
        dataset.loc[ dataset['Weight'] == 0, 'Weight'] = "low"
        dataset.loc[ dataset['Weight'] == 1, 'Weight'] = "mid"
        dataset.loc[ dataset['Weight'] == 2, 'Weight'] = "high"

    if 'FCVC' in modified_data.columns:
        # Converting 'FCVC' from float to int
        modified_data["FCVC"] = modified_data["FCVC"].astype(int)
        modified_data.loc[modified_data["FCVC"] == 1, "FCVC"] = "Rarely"
        modified_data.loc[modified_data["FCVC"] == 2, "FCVC"] = "Sometimes"
        modified_data.loc[modified_data["FCVC"] == 3, "FCVC"] = "Always"

    if 'NCP' in modified_data.columns:
        # Round decimal values to nearest integer (1,2, or 3) 
        modified_data["NCP"] = round(modified_data["NCP"])

        # Convert 'NCP' from float to int
        modified_data["NCP"] = modified_data["NCP"].astype(int)

    if 'CH2O' in modified_data.columns:
        #round decimal values to nearest integer (1,2, or 3)
        modified_data["CH2O"] = round(modified_data["CH2O"])

        # converting 'CH2O' from float to int
        modified_data["CH2O"] = modified_data["CH2O"].astype(int)

        modified_data.loc[modified_data["CH2O"] == 1, "CH2O"] = "Rarely"
        modified_data.loc[modified_data["CH2O"] == 2, "CH2O"] = "Sometimes"
        modified_data.loc[modified_data["CH2O"] == 3, "CH2O"] = "Always"

    if 'FAF' in modified_data.columns:
        # Round decimal values to nearest integer (0, 1, 2, or 3)
        modified_data["FAF"] = round(modified_data["FAF"])

        # converting 'FAF' from float to int
        modified_data["FAF"] = modified_data["FAF"].astype(int)

        modified_data.loc[modified_data["FAF"] == 0, "FAF"] = "Never"
        modified_data.loc[modified_data["FAF"] == 1, "FAF"] = "Rarely"
        modified_data.loc[modified_data["FAF"] == 2, "FAF"] = "Sometimes"
        modified_data.loc[modified_data["FAF"] == 3, "FAF"] = "Always"

    if 'TUE' in modified_data.columns:
        # Round decimal values to nearest integer (1,2, or 3)
        modified_data["TUE"] = round(modified_data["TUE"])

        # Convert 'TUE' from float to int
        modified_data["TUE"] = modified_data["TUE"].astype(int)

        modified_data.loc[modified_data["TUE"] == 0, "TUE"] = "Rarely"
        modified_data.loc[modified_data["TUE"] == 1, "TUE"] = "Sometimes"
        modified_data.loc[modified_data["TUE"] == 2, "TUE"] = "Always"

    # Display the modified DataFrame
    st.write("Modified DataFrame:")
    st.write(modified_data)
    st.write('**Data Transformation:**')

    with st.expander("Removed the following columns from DataFrame"):
        if columns_to_remove:
            for columns in columns_to_remove:
                st.write(f"- {columns}")
        else:
            st.write("No columns selected for removal.")

    with st.expander("Categorized the \'Age\' column into three age groups:"):
        st.write(" - **Age 14 to 21:** Teens")
        st.write(" - **Age 22 to 55:** Adults")
        st.write(" - **Age 56 and above:** Elderly")

    with st.expander("Categorized the \'Height\' column into three height groups:"):
        st.write(" - **Height 1.4 to 1.6:** Low")
        st.write(" - **Height 1.6 to 1.8:** Mid")
        st.write(" - **Height 1.8 and above:** High")

    st.divider()

    return modified_data


def analyze_data(modified_data, uploaded_data):
    # Check for obesity level existence
    obesity_lvl_exist = check_column_existence(uploaded_data, 'Obesity_Level')

    # If obesity level is given, run data analysis against obesity level
    if obesity_lvl_exist:
        analysis_type = st.selectbox("Select Analysis Type", ["Gender Analysis", "Age Analysis", "Height Analysis", 
                        "Weight Analysis", "Obesity Level Analysis", "Family History Analysis", "FAVC Analysis", 
                        "FCVC Analysis", "NCP Analysis", "CAEC Analysis", "SMOKE Analysis", "CH2O Analysis", 
                        "SCC Analysis", "FAF Analysis", "TUE Analysis", "CALC Analysis", "MTRANS Analysis"])

        if analysis_type == "Gender Analysis":
            gender_analysis(uploaded_data)
        elif analysis_type == "Age Analysis":
            age_analysis(uploaded_data, modified_data)
        elif analysis_type == "Height Analysis":
            height_analysis(uploaded_data, modified_data)
        elif analysis_type == "Weight Analysis":
            weight_analysis(uploaded_data)
        elif analysis_type == "Obesity Level Analysis":
            obesity_lvl_analysis(uploaded_data)
        elif analysis_type == "Family History Analysis":
            fam_hist_analysis(modified_data)
        elif analysis_type == "FAVC Analysis":
            favc_analysis(uploaded_data)
        elif analysis_type == "FCVC Analysis":
            fcvc_analysis(modified_data)
        elif analysis_type == "NCP Analysis":
            ncp_analysis(modified_data)
        elif analysis_type == "CAEC Analysis":
            caec_analysis(modified_data)
        elif analysis_type == "SMOKE Analysis":
            smoke_analysis(modified_data)
        elif analysis_type == "CH2O Analysis":
            ch20_analysis(modified_data)
        elif analysis_type == "SCC Analysis":
            scc_analysis(modified_data)
        elif analysis_type == "FAF Analysis":
            faf_analysis(modified_data)
        elif analysis_type == "TUE Analysis":
            tue_analysis(modified_data)
        elif analysis_type == "CALC Analysis":
            calc_analysis(modified_data)
        elif analysis_type == "MTRANS Analysis":
            mtrans_analysis(modified_data)


def check_column_existence(uploaded_data, column_name):
    # Check if the specified column exists
    column_exists = column_name in uploaded_data.columns

    # Display a message based on the existence of the column
    if column_exists:
        # Return the boolean variable
        return column_exists


def gender_analysis(uploaded_data):
    # Subheader for Gender Analysis
    st.subheader('Gender Analysis')

    # Create & display a Seaborn catplot for gender analysis
    st.write('**Catplot of Gender and Obesity Level Distribution**')
    plt.figure()
    gender_plot = sns.catplot(x="Gender", hue="Obesity_Level",kind='count',
            palette="pastel", edgecolor=".6",
            data=uploaded_data)
    st.pyplot(gender_plot)

    # Explanation for the catplot
    explanation = """
    This catplot illustrates the distribution of Obesity Levels based on gender.
    Each bar represents the count of individuals in different Obesity Levels for Females and Males.
    The various color hues differentiate Obesity Levels within each gender category.
    """
    st.write(explanation)
    st.divider()


def age_analysis(uploaded_data, modified_data):
    # Subheader for Age Analysis
    st.subheader('Age Analysis')

    sns.set(style="whitegrid")

    # Create a Seaborn boxplot for age analysis
    st.write("**Boxplot of Age Distribution**")
    plt.figure()
    age_plot_box = sns.boxplot(y=uploaded_data["Age"])
    st.pyplot(plt.gcf())

    # Explanation for the boxplot
    boxplot_explanation = """
    This boxplot visualizes the distribution of ages in the dataset.
    The box represents the interquartile range (IQR), and the line inside the box is the median.
    Whiskers extend to the minimum and maximum values within 1.5 times the IQR.
    Dots outside the whiskers indicate potential outliers.
    """
    st.write(boxplot_explanation)
    st.text("")

    # Create a Seaborn catplot for age analysis categorized into Teens, Adults & Elderly.
    st.write('**Catplot of Age and Obesity Level Distribution**')
    plt.figure()
    age_plot_cat = sns.catplot(x="Age", hue="Obesity_Level",kind='count',
            palette="pastel", edgecolor=".6", data=modified_data)
    st.pyplot(age_plot_cat)

    # Explanation for the catplot
    catplot_explanation = """
    This catplot illustrates the distribution of Obesity Levels across different age groups.
    Each bar represents the count of individuals in different Obesity Levels for each age group.
    The various color hues differentiates Obesity Levels within each age group.
    """
    st.write(catplot_explanation)
    st.divider()


def height_analysis(uploaded_data, modified_data):
    # Subheader for Height Analysis
    st.subheader('Height Analysis')

    sns.set(style="whitegrid")

    # Create a Seaborn boxplot for height analysis
    st.write('**Boxplot of Height Distribution**')
    plt.figure()
    height_plot_box = sns.boxplot(y=uploaded_data["Height"])
    st.pyplot(plt.gcf())

    # Explanation for the boxplot
    boxplot_explanation = """
    This boxplot visualizes the distribution of patients' height in the dataset.
    The box represents the interquartile range (IQR), and the line inside the box is the median.
    Whiskers extend to the minimum and maximum values within 1.5 times the IQR.
    Any dots outside the whiskers may indicate potential outliers in the height distribution.
    """
    st.write(boxplot_explanation)
    st.text("")

    # Create a Histogram for height analysis
    st.write('**Histogram of Height Distribution**')
    plt.figure()
    uploaded_data['Height'].hist(bins=5)
    plt.xlabel("Height in metres")
    plt.ylabel("Number of patients")
    st.pyplot(plt.gcf())

    # Explanation for the histogram
    histogram_explanation = """
    This histogram visualizes the distribution of patients' height in the dataset. 
    The data is divided into five bins, and the histogram displays the number of patients falling into each height range. 
    The x-axis represents height in meters, and the y-axis represents the count of patients in each bin.
    """
    st.write(histogram_explanation)
    st.text("")

    # Create a Seaborn catplot for height analysis categorized into High, Mid & Low.
    st.write('**Catplot of Height and Obesity Level Distribution**')
    plt.figure()
    height_plot_cat = sns.catplot(x="Height", hue="Obesity_Level",kind='count',
            palette="pastel", edgecolor=".6", data=modified_data)
    st.pyplot(height_plot_cat)

    # Explanation for the catplot
    catplot_explanation = """
    This catplot illustrates the distribution of patients' height across different height groups.
    Each bar represents the count of individuals in different Obesity Levels for each height group.
    The various color hues differentiates Obesity Levels within each height group.
    """
    st.write(catplot_explanation)
    st.divider()


def weight_analysis(uploaded_data):
    # Subheader for Weight Analysis
    st.subheader('Weight Analysis')

    sns.set(style="whitegrid")

    # Create a Seaborn boxplot for height analysis
    st.write('**Boxplot of Weight Distribution**')
    plt.figure()
    weight_plot_box = sns.boxplot(y=uploaded_data["Weight"])
    st.pyplot(plt.gcf())

    # Explanation for the boxplot
    boxplot_explanation = """
    This boxplot visualizes the distribution of patients' weight in the dataset.
    The box represents the interquartile range (IQR), and the line inside the box is the median.
    Whiskers extend to the minimum and maximum values within 1.5 times the IQR.
    Any dots outside the whiskers may indicate potential outliers in the weight distribution.
    """
    st.write(boxplot_explanation)
    st.text("")

    # Create a Histogram for weight analysis
    st.write('**Histogram of Weight Distribution**')
    plt.figure()
    uploaded_data['Weight'].hist(bins=6)
    plt.xlabel("Weight in kilograms")
    plt.ylabel("Number of patients")
    st.pyplot(plt.gcf())

    # Explanation for the histogram
    histogram_explanation = """
    This histogram visualizes the distribution of patients' weight in the dataset. 
    The data is divided into six bins, and the histogram displays the number of patients falling into each weight range. 
    The x-axis represents weight in kilograms, and the y-axis represents the count of patients in each bin.
    """
    st.write(histogram_explanation)
    st.divider()


def obesity_lvl_analysis(uploaded_data):
    # Subheader for Obesity Level Analysis
    st.subheader('Obesity Level Analysis')

    # Create a Seaborn catplot for obesity level analysis.
    st.write('**Catplot of Weight and Obesity Level Distribution**')
    plt.figure()
    sns.set(font_scale=0.8)
    obesity_lvl_plot_cat = sns.catplot(data=uploaded_data, x="Obesity_Level", y="Weight", hue="Obesity_Level", kind="box", order=['Insufficient_Weight','Normal_Weight','Overweight_Level_I',
              'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']).set_xticklabels(rotation=15)
    st.pyplot(obesity_lvl_plot_cat)

    # Explanation for the catplot
    catplot_explanation = """
    This catplot visually represents the distribution of weights across different obesity levels.
    Each boxplot represents a specific obesity level, displaying the distribution of weights within that category.
    The x-axis represents different obesity levels, and the y-axis represents weights.
    The data is categorized into:
     - 'Insufficient Weight'
     - 'Normal Weight'
     - 'Overweight Level I'
     - 'Overweight Level II'
     - 'Obesity Type I'
     - 'Obesity Type II'
     - 'Obesity Type III'
    """
    st.write(catplot_explanation)
    st.divider()


def fam_hist_analysis(uploaded_data):
    # Subheader for Family History Analysis
    st.subheader('Family History Analysis')

    # Create a Seaborn distplot for family history analysis.
    st.write('**Distplot of Family History Distribution**')
    plt.figure()
    fam_hist_plot_dist = sns.displot(uploaded_data, x="fam_hist_over-wt", hue="Gender", multiple="dodge")
    st.pyplot(fam_hist_plot_dist)

    # Create a Seaborn distplot for family history analysis.
    st.write('**Catplot of Family History and Obesity Level Distribution**')
    plt.figure()
    fam_hist_plot_cat = sns.catplot(x="fam_hist_over-wt", hue="Obesity_Level",kind='count',
                        palette="pastel", edgecolor=".6", data=uploaded_data)
    st.pyplot(fam_hist_plot_cat)
    st.divider()


def favc_analysis(uploaded_data):
    # Subheader for FAVC Analysis
    st.subheader('FAVC Analysis')

    # Create a Seaborn distplot for FAVC Analysis.
    st.write('**Distplot of FAVC Distribution**')
    plt.figure()
    favc_plot_dist = sns.displot(uploaded_data, x="FAVC", hue="Gender", multiple="dodge")
    st.pyplot(favc_plot_dist)

    # Create a Seaborn catplot for FAVC analysis.
    st.write('**Catplot of FAVC and Obesity Level Distribution**')
    plt.figure()
    favc_plot_cat = sns.catplot(x="FAVC", hue="Obesity_Level",kind='count',
                    palette="pastel", edgecolor=".6", data=uploaded_data)
    st.pyplot(favc_plot_cat)
    st.divider()


def fcvc_analysis(modified_data):
    # Subheader for FCVC Analysis
    st.subheader('FCVC Analysis')

    # Create a Seaborn catplot for FCVC Analysis.
    st.write('**Distplot of FCVC Distribution**')
    plt.figure()
    fcvc_plot_cat1 = sns.catplot(data=modified_data, x="FCVC", kind="count")
    st.pyplot(fcvc_plot_cat1)

    # Create a 2nd Seaborn catplot for FCVC analysis.
    st.write('**Catplot of FCVC and Obesity Level Distribution**')
    plt.figure()
    favc_plot_cat2 = sns.catplot(x="FCVC", hue="Obesity_Level",kind='count',
            palette="pastel", edgecolor=".6", data=modified_data)
    st.pyplot(favc_plot_cat2)
    st.divider()


def ncp_analysis(modified_data):
    # Subheader for NCP Analysis
    st.subheader('NCP Analysis')

    # Create a Seaborn catplot for NCP Analysis.
    st.write('**Catplot of NCP Distribution**')
    plt.figure()
    ncp_plot_cat = sns.catplot(y="NCP", hue="Obesity_Level",kind='count',
            palette="pastel", edgecolor=".6", data=modified_data)
    st.pyplot(ncp_plot_cat)
    st.divider()


def caec_analysis(modified_data):
    # Subheader for CAEC Analysis
    st.subheader('CAEC Analysis')

    # Create a Seaborn countplot for CAEC Analysis.
    st.write('**Countplot of CAEC Distribution**')
    plt.figure()
    sns.countplot( x='CAEC', data=modified_data, palette="Set1");
    st.pyplot(plt.gcf())

    # Create a Seaborn catplot for CAEC Analysis.
    st.write('**Catplot of CAEC and Obesity Level Distribution**')
    plt.figure()
    caec_plot_cat = sns.catplot(y="CAEC", hue="Obesity_Level",kind='count',
            palette="pastel", edgecolor=".6", data=modified_data)
    st.pyplot(caec_plot_cat)
    st.divider()


def smoke_analysis(modified_data):
    # Subheader for SMOKE Analysis
    st.subheader('SMOKE Analysis')

    sns.set(style="darkgrid")

    # Create a Seaborn countplot for SMOKE Analysis.
    st.write('**Countplot of SMOKE Distribution**')
    plt.figure()
    sns.countplot( x='SMOKE', data=modified_data, hue="SMOKE", palette="Set2");
    st.pyplot(plt.gcf())

    # Create a 2nd Seaborn countplot for SMOKE and Gender Analysis.
    st.write('**Countplot of SMOKE and Gender Distribution**')
    plt.figure()
    sns.countplot(x ='Gender', hue = "SMOKE", data =modified_data)
    st.pyplot(plt.gcf())

    st.write('**Catplot of SMOKE and Obesity Level Distribution**')
    plt.figure()
    smoke_plot_cat = sns.catplot(x="SMOKE", hue="Obesity_Level",kind='count', 
                    edgecolor=".1", palette="tab20", data=modified_data)
    st.pyplot(smoke_plot_cat)
    st.divider()


def ch20_analysis(modified_data):
    # Subheader for CH20 Analysis
    st.subheader('CH20 Analysis')

    sns.set(style="darkgrid")

    # Create a Seaborn countplot for CH20 Analysis.
    st.write('**Countplot of CH20 Distribution**')
    plt.figure()
    sns.countplot( x='CH2O', data=modified_data, palette="Set1");
    st.pyplot(plt.gcf())

    # Create a 2nd Seaborn countplot for CH20 Analysis.
    st.write('**Countplot of CH20 and Obesity Level Distribution**')
    plt.figure()
    sns.countplot( x='CH2O', data=modified_data, hue="Obesity_Level", palette="Set2");
    st.pyplot(plt.gcf())
    st.divider()


def scc_analysis(modified_data):
    # Subheader for SCC Analysis
    st.subheader('SCC Analysis')

    sns.set(style="darkgrid")

    # Create a Seaborn countplot for SCC Analysis.
    st.write('**Countplot of SCC Distribution**')
    plt.figure()
    sns.countplot( x='SCC', data=modified_data, hue="SCC", palette="pastel");
    st.pyplot(plt.gcf())

    # Create a Seaborn catplot for SCC Analysis.
    st.write('**Catplot of SCC and Obesity Level Distribution**')
    plt.figure()
    scc_plot_cat = sns.catplot(x="SCC", hue="Obesity_Level",kind='count', 
                    edgecolor=".1", palette="Paired", data=modified_data)
    st.pyplot(scc_plot_cat)
    st.divider()


def faf_analysis(modified_data):
    # Subheader for FAF Analysis
    st.subheader('FAF Analysis')

    sns.set(style="darkgrid")

    # Create a Seaborn countplot for FAF Analysis.
    st.write('**Countplot of FAF Distribution**')
    plt.figure()
    sns.countplot( x='FAF', data=modified_data, palette="pastel");
    st.pyplot(plt.gcf())

    # Create a Seaborn catplot for FAF Analysis.
    st.write('**Catplot of FAF and Obesity Level Distribution**')
    plt.figure()
    faf_plot_cat = sns.catplot(x="FAF", hue="Obesity_Level",kind='count', 
                    edgecolor=".1", palette="colorblind", data=modified_data)
    st.pyplot(faf_plot_cat)
    st.divider()


def tue_analysis(modified_data):
    # Subheader for TUE Analysis
    st.subheader('TUE Analysis')

    sns.set(style="darkgrid")

    # Create a Seaborn catplot for TUE Analysis.
    st.write('**Catplot of TUE and Obesity Level Distribution**')
    plt.figure()
    tue_plot_cat = sns.catplot(x="TUE", hue="Obesity_Level",kind='count', 
                    edgecolor=".1", palette="cubehelix", data=modified_data)
    st.pyplot(tue_plot_cat)
    st.divider()


def calc_analysis(modified_data):
    # Subheader for CALC Analysis
    st.subheader('CALC Analysis')

    sns.husl_palette(8)

    # Create a Seaborn catplot for CALC Analysis.
    st.write('**Catplot of CALC and Obesity Level Distribution**')
    plt.figure()
    calc_plot_cat = sns.catplot(x="CALC", hue="Obesity_Level",kind='count', 
                    edgecolor=".1", data=modified_data)
    st.pyplot(calc_plot_cat)
    st.divider()


def mtrans_analysis(modified_data):
    # Subheader for MTRANS Analysis
    st.subheader('MTRANS Analysis')

    sns.husl_palette(8)

    # Create a Seaborn catplot for MTRANS Analysis.
    st.write('**Catplot of MTRANS and Obesity Level Distribution**')
    plt.figure()
    mtrans_plot_cat = sns.catplot(y="MTRANS", hue="Obesity_Level",kind='count',
            palette="husl", edgecolor=".5", data=modified_data)
    st.pyplot(mtrans_plot_cat)
    st.divider()


def categorize_obesity(modified_data):
    if 'Obesity_Level' in modified_data.columns:
        for dataset in [modified_data]:
            dataset.loc[(dataset['Obesity_Level'] == 'Insufficient_Weight') , 'Obesity_Level'] = 0
            dataset.loc[(dataset['Obesity_Level'] == 'Normal_Weight'), 'Obesity_Level'] = 1
            dataset.loc[(dataset['Obesity_Level'] == 'Overweight_Level_I'), 'Obesity_Level'] = 2
            dataset.loc[(dataset['Obesity_Level'] == 'Overweight_Level_II'), 'Obesity_Level'] = 3
            dataset.loc[(dataset['Obesity_Level'] == 'Obesity_Type_I'), 'Obesity_Level'] = 4
            dataset.loc[(dataset['Obesity_Level'] == 'Obesity_Type_II'), 'Obesity_Level'] = 5
            dataset.loc[(dataset['Obesity_Level'] == 'Obesity_Type_III'), 'Obesity_Level'] = 6

            dataset['Obesity_Level'].astype(int)
            dataset.loc[ dataset['Obesity_Level'] == 0, 'Obesity_Level'] = 0
            dataset.loc[ dataset['Obesity_Level'] == 1, 'Obesity_Level'] = 0
            dataset.loc[ dataset['Obesity_Level'] == 2, 'Obesity_Level'] = 1
            dataset.loc[ dataset['Obesity_Level'] == 3, 'Obesity_Level'] = 1
            dataset.loc[ dataset['Obesity_Level'] == 4, 'Obesity_Level'] = 1
            dataset.loc[ dataset['Obesity_Level'] == 5, 'Obesity_Level'] = 1
            dataset.loc[ dataset['Obesity_Level'] == 6, 'Obesity_Level'] = 1

    return modified_data


def encode_data(modified_data_agn):
    columns_to_remove = [0, 16, 12, 10, 11, 7]
    if columns_to_remove:
        # Remove selected columns from the DataFrame
        modified_data_agn.drop(columns=columns_to_remove, axis=1, inplace=True)
    
    with st.expander("Removed the following columns from DataFrame"):
        if columns_to_remove:
            for columns in columns_to_remove:
                st.write(f"- {columns}")

    st.write(modified_data_agn)

    # Get columns to encode
    listcolumns = modified_data_agn.columns.tolist()
    patientdf = pd.get_dummies(modified_data_agn,columns = listcolumns)
    st.write(patientdf)
    st.divider()

    return patientdf


def split_data(patientdf):
    y = patientdf["Obesity_Level"].values
    x = patientdf.drop(["Obesity_Level"],axis=1).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state = 129)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    return x_train, x_test, y_train, y_test


def decision_tree(x_train, y_train, x_test1, y_test, x_te):
    # Subheader for Decision Tree Model
    st.subheader('Decision Tree Model')

    # Decision Tree Classifier
    from sklearn import datasets
    from sklearn import metrics
    from sklearn.tree import DecisionTreeClassifier

    # fit a dtmodel model to the data
    dtmodel = DecisionTreeClassifier(max_depth=13,splitter="random",
        min_samples_split=2,  # This parameter sets the minimum number of samples required to split an internal node. It controls how fine-grained the splitting process is.
                              # A smaller value can lead to a more complex tree, while a larger value can make the tree more general.
        min_samples_leaf=1,   # The "min_samples_leaf" parameter sets the minimum number of samples required to be in a leaf node.
                              # It controls the size of the leaf nodes. A smaller value can make the tree more detailed, while a larger value can make the tree less complex.
    )
    dtmodel.fit(x_train, y_train)
    # st.write("Max Depth={}".format(dtmodel.tree_.max_depth))

    # make predictions
    expected = y_train
    predicted = dtmodel.predict(x_train)

    # summarize the fit of the model
    st.write("**Decision Tree Model Training Results:**")
    st.code(metrics.classification_report(expected, predicted))
    st.write("**Confusion Matrix:**")
    st.code(metrics.confusion_matrix(expected, predicted))

    # score model for train set
    y_dt= dtmodel.predict(x_train)

    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import mean_squared_error as mse
    from math import sqrt

    # Accuracy score for train set
    from sklearn.metrics import accuracy_score
    score_dt = accuracy_score(y_train, y_dt)
    st.code("Accuracy score for the train set = {:.2f}%".format(score_dt*100))

    precision_dt = metrics.precision_score(y_train, y_dt)
    st.code("Precision score for the train set = {:.2f}%".format(precision_dt*100))

    recall_dt = metrics.recall_score(y_train, y_dt)
    st.code("Recall score for the train set = {:.2f}%".format(recall_dt*100))

    specificity_dt = metrics.recall_score(y_train, y_dt, pos_label=0)
    st.code("Specificity score for the train set = {:.2f}%".format(specificity_dt*100))

    F1_score_dt = metrics.f1_score(y_train, y_dt)
    st.code("F1 score for the train set = {:.2f}%".format(F1_score_dt*100))

    mae_dt = mae(y_train, y_dt)
    st.code("Mean Absolute Error (MAE) for the train set = {:.2f}".format(mae_dt))

    mse_dt = mse(y_train, y_dt)
    st.code("Mean Squared Error (MSE) for the train set = {:.2f}".format(mse_dt))
    st.divider()

    # Test Set
    if '2_Elderly' not in x_test1.columns:
        x_test1.insert(loc=3, column='2_Elderly', value=False)
    y_dttest = dtmodel.predict(x_test1)

    # summarize the fit of the model
    st.write("**Decision Tree Model Testing Results:**")
    st.code(metrics.classification_report(y_test, y_dttest))
    st.write("**Confusion Matrix:**")
    st.code(metrics.confusion_matrix(y_test, y_dttest))

    # Accuracy score for test set
    from sklearn.metrics import accuracy_score
    score_dt = accuracy_score(y_test, y_dttest)
    st.code("Accuracy score for the test set = {:.2f}%".format(score_dt*100))

    precision_dttest = metrics.precision_score(y_test, y_dttest)
    st.code("Precision score for the test set = {:.2f}%".format(precision_dttest*100))

    recall_dttest = metrics.recall_score(y_test, y_dttest)
    st.code("Recall score for the test set = {:.2f}%".format(recall_dttest*100))

    specificity_dttest = metrics.recall_score(y_test, y_dttest, pos_label=0)
    st.code("Specificity score for the test set = {:.2f}%".format(specificity_dttest*100))

    F1_score_dttest = metrics.f1_score(y_test, y_dttest)
    st.code("F1 score for the test set = {:.2f}%".format(F1_score_dttest*100))

    mae_dttest = mae(y_test, y_dttest)
    st.code("Mean Absolute Error (MAE) for the test set = {:.2f}".format(mae_dttest))

    mse_dttest = mse(y_test, y_dttest)
    st.code("Mean Squared Error (MSE) for the test set = {:.2f}".format(mse_dttest))

    rmse_dttest = sqrt(mse_dttest)
    st.code("Root Mean Squared Error (RMSE) for the test set = {:.2f}".format(rmse_dttest))

    miracle_drug = []

    for x in y_dttest:
        if x == 1:
            miracle_drug.append("Yes")
        else:
            miracle_drug.append("No")

    x_te["To give?"] = miracle_drug
    st.write("**Prediction Test Results**")
    st.write(x_te)
    st.divider()


if __name__ == "__main__":
    main()
