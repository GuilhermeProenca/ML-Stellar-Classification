import streamlit as st
import pandas as pd
import numpy as np
import joblib

from PIL import Image

st.title("Projeto Classificador Estelar")
st.text("Objetivo: Classificação de estrelas, galáxias e quasares utilizando Machine Learning.")

u = st.number_input("U - Filtro ultravioleta no sistema fotométrico. (Ex: 23.87882)", format="%f")
g = st.number_input("G - Filtro verde no sistema fotométrico. (Ex: 22.27530)", format="%f")
r = st.number_input("R - Filtro vermelho no sistema fotométrico. (Ex: 20.39501)", format="%f")
i = st.number_input("I - Filtro de infravermelho próximo no sistema fotométrico. (Ex: 19.16573)", format="%f")
z = st.number_input("Z - Filtro infravermelho no sistema fotométrico. (Ex: 18.79371)", format="%f")
spec_obj_ID = st.number_input("Spec_obj_ID - ID exclusivo usado para objetos espectroscópicos ópticos. (Ex: 6.543777e+18)", format="%f")
redshift = st.number_input("Redshift - Valor de desvio para o vermelho baseado no aumento do comprimento de onda. (Ex: 0.634794)", format="%f")
plate = st.number_input("Placa - ID da placa. (Ex: 5812)", format="%f")
MJD = st.number_input("MJD - Modified Julian Date, usada para indicar quando um dado do SDSS foi obtido. (Ex: 56251)", format="%f")

images = ["figs/galaxia.jpg", "figs/estrela.webp", "figs/quasar.webp"]

information = ["A Via Láctea é uma galáxia espiral, da qual o Sistema Solar faz parte.\nVista da Terra, aparece como uma faixa brilhante e difusa que\ncircunda toda a esfera celeste, recortada por nuvens moleculares\nque lhe conferem um intrincado aspecto irregular e recortado.",
               "VY Canis Majoris também conhecida como VY Cma, é uma estrela hipergigante\ncom brilho avermelhado, sendo 2.100 vezes maior que o Sol\nem diâmetro. Para se ter ideia de sua magnitude, dentro dela caberia\nquase três bilhões de planetas iguais à Terra.",
               "TON 618 é um quasar muito distante e extremamente luminoso localizado\npróximo ao Pólo Norte Galáctico na constelação de Canes Venatici\ncom tamanho de 66 bilhões massas solares."
              ]

model = joblib.load('model.pkl')

def predicao(u, g, r, i, z, spec_obj_ID, redshift, plate, MJD):
  
  X = np.reshape([u, g, r, i, z, spec_obj_ID, redshift, plate, MJD], (1,-1))
  pred = model.predict(X)

  if pred == 0:
    text = 'Galáxia'
    img = images[0]
    info = information[0]
  elif pred == 1:
    text = 'Estrela'
    img = images[1]
    info = information[1]
  else:
    text = 'Quasar'
    img = images[2]
    info = information[2]

  return text, img, info

text1, img, info = predicao(u, g, r, i, z, spec_obj_ID, redshift, plate, MJD)

if st.button("Predict"): 
    st.success('Objeto detectado: {}\n'.format(text1))
    image = Image.open(img)
    st.image(image)
    st.markdown('<h4 style="text-align: center; color: black;">"{}"</h4>'.format(info), unsafe_allow_html=True)
