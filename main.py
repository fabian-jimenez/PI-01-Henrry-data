#importe de librerias 
import re 
from fastapi import FastAPI
import pandas as pd 
import numpy as np 
import uvicorn 
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# creacions de la api 

app = FastAPI (title= 'API proyecto MOVIES para henrry bootcamp')

# lectura de csv del ETL 
dfmovies=pd.read_csv('Movies_final_ETL.csv')


#consulta de funcionamiendo de la api 
@app.get ('/')

def read_root():
    return {"hola fabian"}


#consulta  de cuantas peliculas se ingresaron ese mes 
 
@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes:int):
    '''Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes historicamente'''
    dfmovies['release_date']= pd.to_datetime(dfmovies['release_date'])
    peliculas_mes = dfmovies [dfmovies['release_date'].dt.month == mes] # filtra por mes 
    respuesta = len(peliculas_mes) #cuenta la cantidad de peliculas por mes 
    return {'mes':mes, 'cantidad':respuesta}

@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia:int):
    '''Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrebaron ese dia historicamente'''
    dfmovies['release_date']=pd.to_datetime(dfmovies['release_date'])
    peliculas_dia =dfmovies[dfmovies['release_date'].dt.day == dia ] # filtra por dia 
    respuesta = len(peliculas_dia) #cuenta la cantidad por dia 
    return {'dia':dia, 'cantidad':respuesta}

@app.get('/score_titulo/{titulo}')
def score_titulo(titulo:str):
    '''Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score'''
    titulo_limpio= re.sub(r'\[.*\]','',titulo)#limpiar el titulo 
    titulo_limpio = titulo_limpio.strip() #elimina espacios antes y alfinal 
    dfmovies_limpio = dfmovies.dropna(subset=['belongs_to_collection'])# alimina datos NAN

    pelicula_titulo = dfmovies_limpio[dfmovies_limpio['belongs_to_collection'].str.contains(titulo_limpio, case=False)] #filtra por titulo 
    
    if pelicula_titulo.empty: #veridficacion de titulo 
       return{'Titulo no encontrado'}
    titulo= pelicula_titulo['belongs_to_collection'].values[0] # guarda informacion de esa coluna en una variable 
    año = pelicula_titulo['release_year'].values[0]
    popularidad= pelicula_titulo['popularity'].values[0]

    return {'titulo':titulo, 'año':año, 'popularidad':popularidad}

@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo:str):
    '''Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones. 
     La misma variable deberá de contar con al menos 2000 valoraciones, 
   caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.'''
    titulo_limpio= re.sub(r'\[.*\]','',titulo)#limpiar el titulo 
    titulo_limpio = titulo_limpio.strip() #elimina espacios antes y alfinal 
    dfmovies_limpio = dfmovies.dropna(subset=['belongs_to_collection'])# alimina datos NAN   
    
    pelicula_titulo = dfmovies_limpio[dfmovies_limpio['belongs_to_collection'].str.contains(titulo_limpio, case=False)] #filtra por titulo  
    
    if pelicula_titulo.empty: #veridficacion de titulo 
       return{'Titulo no encontrado'}
    
    titulo= pelicula_titulo['belongs_to_collection'].values[0] # guarda informacion de esa coluna en una variable 
    año = pelicula_titulo['release_year'].values[0]
    voto_total =pelicula_titulo['vote_count'].values[0]
    if voto_total <= 2000 : #condicion de menos a 2000
        return['La pelicula no cumple con almenos 2000 valoraciones']
    
    voto_promedio = pelicula_titulo['vote_average'].values[0]

    return {'titulo':titulo, 'año':año, 'voto_total':voto_total, 'voto_promedio':voto_promedio }

@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor:str):
    '''Se ingresa el nombre de un actor que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
    Además, la cantidad de películas que en las que ha participado y el promedio de retorno'''
    peliculas_actor= dfmovies[dfmovies['cast'].str.contains(nombre_actor , case=False, na=False)] #filtar por actor 
    if peliculas_actor.empty: # verificacion del actor 
        return{'Actor no encontrado'}
    cantidad_actor=len(peliculas_actor)
    retorno_total= peliculas_actor['return'].values[0] #guarda informacion de la columna de retur del fultro anterior 
     
    return {'actor':nombre_actor, 'cantidad_filmaciones':cantidad_actor, 'retorno_total':retorno_total}

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):
    ''' Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. 
    Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.'''
   
    peliculas_director= dfmovies[dfmovies['crew'].str.contains(nombre_director, case=False , na=False)] #filtrar por el nombre del director 
    if peliculas_director.empty: #verificacion del Director 
        return{'Director no encontrado'}
    cantidad_peliculas_director=[] #lista bacia para almasenar cada pelicula del director con su informacion 
    for _, row in peliculas_director.iterrows(): #agregando a la lista del director 
        cantidad_peliculas_director_2 = {
            'Nombre de la pelicula': row['title'],
            'fecha de lanzamiento ' : row['release_date'],
            'Retorno': row['return'] if not math.isinf(row['return'])and np.isfinite(row['return']) else 0, # retorno del valor con condicional de valores nulos e infinitos 
        }   
        cantidad_peliculas_director.append(cantidad_peliculas_director_2)#funcion de agregacion 


    return {'director':nombre_director,'peliculas':cantidad_peliculas_director}

# ML
@app.get('/recomendacion/{titulo}')

def recomendacion(titulo:str):
#    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    tfidf= TfidfVectorizer
    tfidf_matrix = tfidf.fit_transform(dfmovies['title'].values.astype('U'))
    similitudes = linear_kernel(tfidf_matrix,tfidf_matrix)

    def recomendaciones_movies (movie_title, similitudes ,dfmovies, top_n =5 ) :
        movie_index = dfmovies[dfmovies['title'] == movie_title].index[0]
        similarity_scores = list(enumerate(similitudes[movie_index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_movies_indices = [i[0] for i in similarity_scores[1:top_n+1]]
        top_movies = dfmovies['title'].iloc[top_movies_indices]
        return list(top_movies)
        
    top_movies= recomendaciones_movies(titulo, similitudes , dfmovies)
    return {'lista recomendada': top_movies}

uvicorn.run(app , host="192.168.18.148",port=8000)