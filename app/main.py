import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data.drop(['id','Unnamed: 32'], axis=1, inplace=True)
    data['diagnosis']= data['diagnosis'].map({'M':1, 'B':0})

    return data


def add_sidebar():
    st.sidebar.header('Cell Nuclei Measurements')

    data = get_clean_data()
    input_dict = {}

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    for label, key in slider_labels:
        input_dict[key]=st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )  
    return input_dict  


def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict


def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']
   
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible = True,
                range=[0, 1]
            )
        ),
        showlegend = True
    )
    
    return fig

def add_predictions(input_data):
    model = pickle.load(open('model/modelfit.pkl', 'rb'))
    scaler = pickle.load(open('model/Scaler.pkl', 'rb'))

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    scaled_model = scaler.transform(input_array)

    predictioin = model.predict(scaled_model)
 
    st.subheader('Cancer cell cluster prediction')
    st.write('The cell cluster is:')

    if predictioin[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class= 'diagnosis malicious'> Malicious </span>", unsafe_allow_html=True)
    
    st.write('Probability of being benign: ', model.predict_proba(scaled_model)[0][0])
    st.write('Probability of being Maliciuos: ', model.predict_proba(scaled_model)[0][1])

    st.write("While this software can help doctors diagnose patients, it shouldn't be used in place of a professional diagnostic.")


#User profile
username = 'Lamda'
contact1 = '''
LinkedIn: <i>Link</i>  <br> 
Email: tetteycollins@gmail.com'''

url = "www.linkedin.com/in/tettey-collins-kwabena-10351332a"
contact = "LinkedIn Profile:  [Link](%s)" % url

imagepath = r'assets/images/croped.PNG'

def user_card(username, contact, imagepath):
    img_col, contact_col = st.columns([0.7,4])
    
    with img_col:
        st.image(imagepath, caption='Profile Picture')
    with contact_col:
        st.write('Username: ', username)
        st.markdown(contact, unsafe_allow_html=True)
    


def main():
    st.set_page_config(
        page_title= 'Breast Cancer Predictor',
        page_icon= r'assets/icons/icons8_pink_ribbon_48.png',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    with open(r'assets/style/style.css') as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
       
    user_card(username, contact, imagepath)
   
    input_data = add_sidebar()
    
    profile = st.container()

    with st.container():
        st.title('Breast Cancer Predictor')
        st.write("Please link this app to your cytology lab so that it can use your tissue sample to identify breast cancer. Based on data from your cytosis lab, this app uses a machine learning algorithm to predict if a breast lump is benign or cancerous. Using the sliders in the sidebar, you can manually update the measurements as well. ")
    
    col1, col2 = st.columns([4,1])
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)
 


if __name__=='__main__':
    main()