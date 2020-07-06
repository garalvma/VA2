import cv2
import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.measure import LineModelND, ransac
import sys

clf = LinearDiscriminantAnalysis()
bayes = cv2.ml.NormalBayesClassifier_create()
nom = sys.argv[1]
resultado = nom.split("/")
f = open(resultado[len(resultado)-1]+".txt", "w")
f.write("nombre x y matricula largo \n")
f.close()
visualizar = False
if len(sys.argv) == 3:
    if sys.argv[2] == 'True':
        visualizar = True

caracteres = ['0','1','2','3','4','5','6','7','8','9','A', 'B', 'C','D','E','ESP', 'F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def numero(arrw):
    index = 0
    maxi = 0
    for i in range(len(arrw)-1):
        if arrw[i] > maxi:
            maxi = arrw[i]
            index = i

    return index

#Funcion de entrenamiento de los clasificadores
def entrenamiento():
    fotos = []

    rootDir = ".\Training_ocr"
    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Directorio encontrado: %s' % dirName)
        for fname in fileList:
            fotos.append(fname)

    caracteristicas = np.zeros((9250,100), dtype=np.float32)
    etiqueta = np.zeros((9250, 1), dtype=np.int32)
    cont = -1
    cont1 = 0
    for u in range(len(fotos)):
        cont3 = 0
        arrx = []
        arry = []
        arrw = []
        arrh = []
        img = cv2.imread(rootDir + "/" + fotos[u], 0)
        bin_img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        contours, hil = cv2.findContours(bin_img, 1, 2)
        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            arrx.append(x)
            arry.append(y)
            arrw.append(w)
            arrh.append(h)

        index = numero(arrw)
        recorte = bin_img[arry[index]:(arry[index] + arrh[index]),arrx[index]:(arrx[index] + arrw[index])]
        resized = cv2.resize(recorte, (10, 10), interpolation=cv2.INTER_LINEAR)

        for i in range(10):
            for j in range(10):
                caracteristicas[cont1][cont3]=resized[i][j]
                cont3 = cont3+1
        if u % 250 == 0:
            cont = cont + 1
        etiqueta[cont1][0]=cont
        cont1 = cont1 + 1

    clf.fit(caracteristicas, etiqueta)
    cr = clf.transform(caracteristicas)
    cr1 = np.asarray(cr, dtype=np.float32)

    bayes.train(cr1, cv2.ml.ROW_SAMPLE, etiqueta)

#Funcion que saca el maximo
def maximo(g, m, n):
    max = 0
    b = 0
    v = 0
    for i in range(m):
        for j in range(n):
            if g[i][j] > max:
                b=i
                v=j
                max=g[i][j]
    return (b,v)

#Funcion que ordena los recuadros segun su coordenada de eje x
def ordenar(ax, ay, aw, ah):
    array5 = []
    for dato in range(len(ax) - 1, 0, -1):
        for i in range(dato):
            if ax[i] > ax[i + 1]:
                temp = ax[i]
                ax[i] = ax[i + 1]
                ax[i + 1] = temp
                temp2 = ay[i]
                ay[i] = ay[i + 1]
                ay[i + 1] = temp2
                temp3 = aw[i]
                aw[i] = aw[i + 1]
                aw[i + 1] = temp3
                temp4 = ah[i]
                ah[i] = ah[i + 1]
                ah[i + 1] = temp4

    for i in range(len(ax)):
        array2 = []
        puntoInicial = (ax[i], ay[i])
        puntoFinal = (aw[i], ah[i])
        array2.append(puntoInicial)
        array2.append(puntoFinal)
        array5.append(array2)
    return array5

#Funcion para mostrar las imagenes con recuadros
def mostrar(imgNormal, array5):
    mitadx = 0
    mitady = 0
    valor = []
    a = 0
    arr1 = []
    for aa in range(len(array5)):
        if len(array5) == 9 and aa == 0:
            print()
        else:
            aa+1
            arr = []
            arr2 = []
            crop_img = bin_img[array5[aa][0][1]:(array5[aa][0][1] + array5[aa][1][1]), array5[aa][0][0]:(array5[aa][0][0] + array5[aa][1][0])]

            recortada = cv2.resize(crop_img, (10, 10), interpolation=cv2.INTER_LINEAR)
            for ae in range(10):
                for j in range(10):
                    arr.append(recortada[ae][j])
            arr2.append(arr)
            recor2 = clf.transform(arr2)
            recor2 = np.asarray(recor2, dtype=np.float32)
            hola1 = bayes.predictProb(recor2)[1]
            indice = hola1[0][0]
            valor = caracteres[indice]
            arr1.append(valor)
            imgNormal = cv2.rectangle(imgNormal, array5[aa][0],(array5[aa][0][0] + array5[aa][1][0], array5[aa][0][1] + array5[aa][1][1]), (0, 255, 0), 2)
            gg = (array5[aa][0][0], (array5[aa][0][1]-10))
            imgNormal= cv2.putText(imgNormal, valor, gg,cv2.FONT_HERSHEY_SIMPLEX,0.65,(255, 255, 255),2)
    if len(array5) != 0:
        imgNormal = cv2.rectangle(imgNormal, ((array5[0][0][0]-15), (array5[0][0][1]-15)),
                                  (((array5[(len(array5)-1)][0][0] + array5[(len(array5)-1)][1][0])+15 ),
                                   ((array5[(len(array5)-1)][0][1] + array5[(len(array5)-1)][1][1])+15)), (0, 100, 255),2)
        mitadx = int(((array5[0][0][0]-15)+(array5[(len(array5)-1)][0][0] + array5[(len(array5)-1)][1][0])+15)/2)
        mitady = int(((array5[0][0][1]-15)+(array5[(len(array5)-1)][0][1] + array5[(len(array5)-1)][1][1])+15)/2)
        a = array5[(len(array5) - 1)][1][0]
        imgNormal = cv2.circle(imgNormal, (mitadx, mitady), 8, (0, 0, 0), 2)
    if visualizar == True:
        cv2.imshow("contornos", imgNormal)
        cv2.waitKey(0)


    return mitadx, mitady, arr1, a

#Algoritmo de RANSAC para filtrar perfectamente la matricula
def ransa(array3):
    array5 = []
    array4 = []
    l = []
    data = np.zeros((len(array3), 2))
    for g in range(len(array3)):
        data[g] = (array3[g][0][0], array3[g][0][1])
        l.append(array3[g][0][0])

    # fit line using all data
    model = LineModelND()
    model.estimate(data)

    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                                   residual_threshold=1, max_trials=1000)
    outliers = inliers == False

    # generate coordinates of estimated models
    line_x = l
    line_y = model.predict_y(line_x)
    line_y_robust = model_robust.predict_y(line_x)

    for t in range(len(array3)):
        for f in range(len(line_x)):
            if abs(array3[t][0][0] - line_x[f]) <= 20 and abs(array3[t][0][1] - line_y_robust[f]) <= 20:
                if array3[t] not in array4:
                    array4.append(array3[t])

    if len(array4) >= 9:
        for po in range(len(array4)):
            if abs(array4[po][0][0] - pixel1) < 200 and abs(array4[po][0][1] - pixel2) < 200:
                array5.append(array4[po])
    else:
        array5 = array4.copy()
    return array5

#Filtra los contornos por dimensiones para quedarnos con los minimos posibles
def filtradogeneral(array1):
    array3 = []
    for i in range(len(array1)):
        array2 = []

        array2.append([(array1[i][0][0], array1[i][0][1]), (array1[i][1][0], array1[i][1][1])])

        for j in range(len(array1)):
            if i != j:
                for k in range(len(array2)):
                    if (float(array1[j][1][0]) / array1[j][1][1]) < 0.85 and (float(array1[j][1][0]) / array1[j][1][1]) > 0.25:
                        if abs(array2[k][0][1] - array1[j][0][1]) < 10:
                            if abs((array2[k][0][0] + array2[k][1][0]) - array1[j][0][0]) < 25:
                                if abs(array2[k][1][1] - array1[j][1][1]) < 5:
                                    if abs(array2[k][1][0] - array1[j][1][0]) < 10:
                                        s = [(array1[j][0][0], array1[j][0][1]), (array1[j][1][0], array1[j][1][1])]
                                        if s not in array2:
                                            array2.append(s)
            if j == len(array1) - 1:
                if len(array2) < 6 or len(array2) > 8:
                    array2 = []

            if len(array2) != 0 and j == len(array1) - 1:
                for ll in range(len(array2)):
                    if array2[ll] not in array3:
                        array3.append(array2[ll])
    return array3

#Saca y dibuja el centro de cada coche
def dibujarcentro(img, imgNormal):
    sift = cv2.ORB_create(100, 1.3, 4, 31, 0, 2, cv2.ORB_HARRIS_SCORE, 31, 20)
    kp1, des1 = sift.detectAndCompute(img, None)
    width, height = img.shape[:2]
    matriz = []
    for i in range(int(height / 10)):
        fila = []
        for j in range(int(width / 10)):
            fila.append(0)
        matriz.append(fila)
    for g in range(len(kp1)):
        x1 = int(kp1[g].pt[0] / 10)
        y1 = int(kp1[g].pt[1] / 10)
        matriz[x1][y1] = matriz[x1][y1] + 1
    (p, m) = maximo(matriz, int(height / 10), int(width / 10))
    pixel1 = p * 10
    pixel2 = m * 10
    imgNormal = cv2.circle(imgNormal, (pixel1, pixel2), 2, (0, 0, 255), 2)
    imgNormal = cv2.rectangle(imgNormal, (pixel1 - 130, pixel2 - 80), (pixel1 + 130, pixel2 + 120), (0, 0, 255), 2)

    return pixel1, pixel2

#Saca todos los contornos de cada imagen
def analisisimagen(rootDir):
    imgNormal = cv2.imread(rootDir + "/" + fotos[u])
    img = cv2.imread(rootDir + "/" + fotos[u], 0)
    bin_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, hil = cv2.findContours(bin_img, 1, 2)
    for i in range(len(contours)):
        array2 = []
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        puntoInicial = (x, y)
        puntoFinal = (w, h)
        array2.append(puntoInicial)
        array2.append(puntoFinal)
        array1.append(array2)
    return imgNormal, array1, img, bin_img

def fichero(mitadx, mitady, nombre, valor, w):
    str1 = ""
    for i in range(len(valor)):
        str1 = str1 + str(valor[i])
    f = open(resultado[len(resultado)-1]+".txt", "a")
    f.write(nombre + " " + str(mitadx) + " " + str(mitady) + " " + str1 + " " + str(int(w/2)) + "\n")
    f.close()


entrenamiento()
fotos = []
rootDir=sys.argv[1]
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Directorio encontrado: %s' % dirName)
    for fname in fileList:
        fotos.append(fname)

for u in range(len(fotos)):
    array1 = []
    array2 = []
    array3 = []
    array4 = []
    array5 = []
    arrayx = []
    arrayy = []
    arrayw = []
    arrayh = []

    imgNormal, array1, img, bin_img = analisisimagen(rootDir)

    pixel1, pixel2 = dibujarcentro(img, imgNormal)

    #Me quedo con los contornos de un determinado tamaño
    for i in range(len(array1)):
        if array1[i][1][1] > 10 and array1[i][1][1] < 45:
            if array1[i][1][0] > 5 and array1[i][1][0] < 25:
                arrayx.append(array1[i][0][0])
                arrayy.append(array1[i][0][1])
                arrayw.append(array1[i][1][0])
                arrayh.append(array1[i][1][1])

    array1 = ordenar(arrayx, arrayy, arrayw, arrayh)

    array3 = filtradogeneral(array1)

    #RANSAC
    if len(array3) != 0:
        array5 = ransa(array3)

    arrayx = []
    arrayy = []
    arrayw = []
    arrayh = []
    #Ordenamos por x los cuadrados de la matricula
    for i in range(len(array5)):
        arrayx.append(array5[i][0][0])
        arrayy.append(array5[i][0][1])
        arrayw.append(array5[i][1][0])
        arrayh.append(array5[i][1][1])

    array5 = ordenar(arrayx, arrayy, arrayw, arrayh)
    mitadx, mitady, valor, w = mostrar(imgNormal, array5)
    fichero(mitadx, mitady, fotos[u], valor, w)

