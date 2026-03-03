# BJJ Position Classifier

## Introducció 
Aquest projecte consisteix en el desenvolupament d'un classificador automàtic de posicions de Brazilian Jiu-Jitsu (BJJ) utilitzant tècniques de visió per computador i anàlisi de pose humana. 

El sistema és capaç d’analitzar una imatge on apareixen dos atletes practicant BJJ i identificar la posició de combat que estan executant. 

Aquest tipus d’eines poden ser útils en l’àmbit de l’anàlisi esportiva, l’entrenament assistit per intel·ligència artificial o la creació de sistemes automàtics d’anotació de combats.

## Objectiu 
L'objectiu principal d'esquest programa és desenvolupar un sistema capaç de classificar automàticament posicions de Brazilian Jiu-Jitsu a partir d'una imatge. És a dir, que donada una imatge en la que apareixen dos atletes, el sistema és capaç de predir la posició del combat que están realitzant. 

Per aquest projecte, es consideren les següents posicions: montada, de peu, derribo, guardia oberta, guardia tancada, mitja guardia, guardia 50-50, control lateral, esquena i tortuga. 


## Dataset
S'ha utilitzat un conjunt de dades que té 120.279 imatges etiquetades de dos atletes de jiu-jitsu entrenant en diferents posicions de combat. Aquestes imatges provenen de 3 càmeres diferents durant 6 seqüències d'sparring, de manera que s'han capturat les posicions desde diversos angles i perspectives. 

Aquests dataset consta de 10 posicions de combat, que resulten en un total de 18 classes, per indicar quin atleta es troba en la posició dominant o inicia l'acció. Les classes són les següents: 
- *standing*: posició de peu. 
- *takedown1, takedown2*: derribo, cada classe indica l'atleta que l'inicia. 
- *open_guard1, open_guard2*: guardia obert. En aquest dataset no es distingeixen entre els diferents tipus de guardia oberta. 
- *half_guard1, half_guard2*: mitja guardia.
- *closed_guard1, closed_guard2*: guardia tancada.
- *5050_guard*: guardia 50-50. En aquesta posició tots dos atletes es troben en una situació simètrica de guardia, sovint amb enredos de cames o doble control. 
- *side_control1, side_control2*: control lateral. Les posicions nort-sur i genoll sobre el ventre es consideren dins d'aquesta categoría.
- *mount1, mount2*: montada. 
- *back1, back2*: esquena. Cada classe indica quin atleta controla l'esquena del oponent. Tot i que en competició normalment es requereixen ganxos amb les cames per considerar aquesta posició, aquesta restricció no s’aplica en aquest dataset.
- *turtle1, turtle2*: posició de tortuga.

Les poses dels atletes es representen utilitzant el format MS-COCO, un estàndard molt usat en visió per computadir per a la detecció de la pose humana. Consta de 17 punts clau (keypoints) que representen les articulacions dels atletes. A més, cada punt clau es representa amb les seves coordenades x i y en la imatge i la confiança del detector: [x, y, c]. 

Totes aquestes anotacions del dataset s'emmagatzemen en una matriu d'objectes JSON. Per tant, cada anotació conté els seguents valors: 
- **imatge**: nom de la imatge .
- **pose1 i pose2**: pose dels atletes en el format MS-COCO ([[x0, y0, c0]… [x16, y16, c16]]).
- **posició**: códi que indica la posició del combat
- **frame**: número de fotograma dins del vídeo original. 

## Model
Arquitectura utilizada (CNN, ResNet, etc.)

## Resultats 
Accuracy, confusion matrix, etc.

## Autor
Lídia Budiós - Universitat de Barcelona - 2026