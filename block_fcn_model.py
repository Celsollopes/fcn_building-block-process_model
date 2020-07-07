import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
#import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from skimage.io import imsave
import skimage as sk
from skimage.io import imread
import imutils
from keras import backend as K
import time
import PIL
from PIL import Image

PATH_TO_FROZEN_GRAPH = 'resnetNewBase_combined-stroke.pb'
count1=0
SAIDA=np.empty((1,512,512,1))
ENTRADA=np.empty((1,512,512,1))
GTIMG=np.empty((1,512,512,1))

### constantes de blocos ###
__DEF_HEIGHT = 512
__DEF_WIDTH = 512
__DEF_WIDTH_NET = 512
__DEF_HEIGHT_NET = 512

"""
    O código está divido para rodar nas duas versões do Tensorflow 1.x e Tensorflow 2.x
"""
if tf.__version__.startswith('1.'):
    print('Tensorflow 1.x')
    #### Run tensorflow version 1.x ####
    with tf.Session() as sess:
        print("Load graph model...")
        with gfile.FastGFile(PATH_TO_FROZEN_GRAPH,'rb') as f:
            graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.compat.v1.import_graph_def(graph_def, name='')
        graph_nodes=[n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
        #print(names)
        # Get handles to # and output tensors
        ops = tf.compat.v1.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}

        tensor_dict = {}
        for key in [
           'refined/Sigmoid' ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                  tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                                  tensor_name)
        image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('input_2:0')

        #### Input da imagem ####
        #imageCinza = cv2.imread("input_image/NewOutput.png", cv2.IMREAD_GRAYSCALE)
        _img = cv2.imread('input_image/katia.png', cv2.IMREAD_GRAYSCALE)
        height, width = _img.shape

        ########## blocks generator #########
        if height > __DEF_HEIGHT or width > __DEF_WIDTH:
            nBlocksH = int(height / __DEF_HEIGHT)
            if __DEF_HEIGHT * nBlocksH < height:
                nBlocksH += 1

            nBlocksW = int(width / __DEF_WIDTH)
            if __DEF_WIDTH * nBlocksW < width:
                nBlocksW += 1

            new_height = (__DEF_HEIGHT * nBlocksH)
            new_width = (__DEF_WIDTH * nBlocksW)
        else:
            new_height = height
            new_width = width

        bckg = Image.new('RGBA',  (new_width, new_height), (0,0,0,255))

        #### Criação dos limites maximos e mínimos da imagem (iniX:endX and iniY:endY) ####

        for j in range(0,nBlocksH):
            iniY = j*__DEF_HEIGHT
            endY = iniY + __DEF_HEIGHT
            if endY > height:
                iniY = iniY - (endY-height)
                endY = height

            '''print('block j = {}'.format(j))
            print('iniY, endY = {},{}'.format(iniY,endY))'''

            for k in range(0,nBlocksW):
                iniX = k*__DEF_WIDTH
                endX = iniX + __DEF_WIDTH
                if endX > width:
                    iniX = iniX - (endX-width)
                    endX = width

                '''print('block k = {}'.format(k))
                print('iniX, endX = {},{}'.format(iniX,endX))'''

                _img_crop = _img[iniY:endY, iniX:endX]
                #regiao = Image.open(_img_crop)
                height_crop, width_crop = _img_crop.shape
                _img_crop = cv2.resize(_img_crop, (__DEF_WIDTH_NET, __DEF_HEIGHT_NET), interpolation=cv2.INTER_CUBIC)
                #print('height_crop, width_crop = {},{}'.format(height_crop,width_crop))
                _img_crop = _img_crop.astype('float32')
                #_img_crop = 1 - (_img_crop.reshape((__DEF_WIDTH_NET, __DEF_HEIGHT_NET, 1)) / 255)
                ## testes de saida ##
                np_regiao = np.array(_img_crop)
                regiao = Image.fromarray(np_regiao)

                bckg.paste(regiao, (iniX, iniY))# inicia a colagem dos blocos a partitr dos indices iniX,iniY

        bckg.save('input_image/NovaImagemCriada.png')# Salva a imagem com adicao do padding
        background = Image.new('RGBA',  (new_width, new_height), (0,0,0,255))

        #### Input da nova imagem (imagem com )com as dimensões dos blocos ####
        print('-'*30)
        #time.sleep(30)
        
        imageCinza_ = cv2.imread("input_image/NovaImagemCriada.png", cv2.IMREAD_GRAYSCALE)
        height_cin, width_cin = imageCinza_.shape

        #gaussian_3 = cv2.GaussianBlur(imageCinza_, (9,9), 10.0)
        #imageCinza = cv2.addWeighted(imageCinza_, 2.0, gaussian_3, -0.5, 0, imageCinza_)
        cv2.imwrite('output/imageWithSharpen.png',imageCinza)

        print("Dimensões da imagem: {}, {}".format(width_cin, height_cin))

        #temp = inputImage
        stepSize =  384# stride - step size in slide window
        (w_width, w_height) = (512, 512)# window size or kernel size
        for y in range(0, imageCinza.shape[1], stepSize):
            for x in range(0, imageCinza.shape[0], stepSize):
                window = imageCinza[x:x + w_width, y:y + w_height]
                #print('Resize window:',window.shape)
                window = cv2.resize(window, (512, 512), interpolation=cv2.INTER_CUBIC)
                window = 1 - (window.reshape((512, 512))/255)

                ENTRADA[0, :, :, 0] = window
                #ENTRADA[0, :, :, 0] = cv2.resize(window (512, 512))
                output_dict = sess.run(tensor_dict,
                                  feed_dict={image_tensor: ENTRADA})
               #print(output_dict['pre#d_a/mul_33'][0])
                ## TRATAMENTO DA SAIDA ##
                SAIDA=output_dict['refined/Sigmoid']
                saida = sk.img_as_ubyte(SAIDA[0, :, :, 0])
                window = sk.img_as_ubyte(window[:])

                #convertendo saida em numpyarray para image
                np_img = np.array(saida)
                #np_img = cv2.resize(np_img, (256, 256))
                img = Image.fromarray(np_img)

                ## CONCATENA BLOCKS ##

                #print('img: ',img.size)
                background.paste(img, (y, x))    
                #cv2.imshow("Output-block", saida)
                #cv2.imwrite("output/image-block-test-0"+str(y)+str(x)+".png", saida)
                background.save('output/saidaImagemBlocos.png')
                
    print('-'*10+'Concluido'+'-'*10)
                
elif tf.__version__.startswith('2.'):
    print('Versão do Tensorflow: ',tf.__version__)
    tf.compat.v1.disable_v2_behavior()
    
    #### Run tensorflow version 2.x ####
    with tf.compat.v1.Session() as sess:
        print("Load graph model...")
        with gfile.FastGFile(PATH_TO_FROZEN_GRAPH,'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        graph_nodes=[n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
        #print(names)
        # Get handles to # and output tensors
        ops = tf.compat.v1.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}

        tensor_dict = {}
        for key in [
           'refined/Sigmoid' ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                  tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                                  tensor_name)
        image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('input_2:0')

        #### Input da imagem ####
        #imageCinza = cv2.imread("input_image/NewOutput.png", cv2.IMREAD_GRAYSCALE)
        _img = cv2.imread('input_image/katia.png', cv2.IMREAD_GRAYSCALE)
        height, width = _img.shape

        ########## Gerador de blocos #########
        if height > __DEF_HEIGHT or width > __DEF_WIDTH:
            nBlocksH = int(height / __DEF_HEIGHT)
            if __DEF_HEIGHT * nBlocksH < height:
                nBlocksH += 1

            nBlocksW = int(width / __DEF_WIDTH)
            if __DEF_WIDTH * nBlocksW < width:
                nBlocksW += 1

            new_height = (__DEF_HEIGHT * nBlocksH)
            new_width = (__DEF_WIDTH * nBlocksW)
        else:
            new_height = height
            new_width = width

        bckg = Image.new('RGBA',  (new_width, new_height), (0,0,0,255))

        #### Criação dos limites maximos e mínimos da immagem (iniX:endX and iniY:endY) ####
        for j in range(0,nBlocksH):
            iniY = j*__DEF_HEIGHT
            endY = iniY + __DEF_HEIGHT
            if endY > height:
                iniY = iniY - (endY-height)
                endY = height

            '''print('block j = {}'.format(j))
            print('iniY, endY = {},{}'.format(iniY,endY))'''

            for k in range(0,nBlocksW):
                iniX = k*__DEF_WIDTH
                endX = iniX + __DEF_WIDTH
                if endX > width:
                    iniX = iniX - (endX-width)
                    endX = width

                '''print('block k = {}'.format(k))
                print('iniX, endX = {},{}'.format(iniX,endX))'''

                _img_crop = _img[iniY:endY, iniX:endX]
                #regiao = Image.open(_img_crop)
                height_crop, width_crop = _img_crop.shape
                _img_crop = cv2.resize(_img_crop, (__DEF_WIDTH_NET, __DEF_HEIGHT_NET), interpolation=cv2.INTER_CUBIC)
                #print('height_crop, width_crop = {},{}'.format(height_crop,width_crop))
                _img_crop = _img_crop.astype('float32')
                #_img_crop = 1 - (_img_crop.reshape((__DEF_WIDTH_NET, __DEF_HEIGHT_NET, 1)) / 255)
                
                ## testes de saida ##
                np_regiao = np.array(_img_crop)
                regiao = Image.fromarray(np_regiao)

                bckg.paste(regiao, (iniX, iniY))# inicia a colagem dos blocos a partitr dos indices iniX,iniY

        bckg.save('input_image/NewOutput.png')
        background = Image.new('RGBA',  (new_width, new_height), (0,0,0,255))

        #### Input da nova imagem com as dimensões dos blocos ####
        print('-'*30)
        #time.sleep(30)
        imageCinza_ = cv2.imread("input_image/NewOutput.png", cv2.IMREAD_GRAYSCALE)
        height_cin, width_cin = imageCinza_.shape

        gaussian_3 = cv2.GaussianBlur(imageCinza_, (9,9), 10.0)
        imageCinza = cv2.addWeighted(imageCinza_, 2.0, gaussian_3, -0.5, 0, imageCinza_)
        cv2.imwrite('output/imageWithSharpen.png',imageCinza)

        print("Dimensões da Imagem: {}, {}".format(width_cin, height_cin))

        #temp = inputImage
        stepSize =  384# stride - step size in slide window
        (w_width, w_height) = (512, 512)# window size or kernel size
        for y in range(0, imageCinza.shape[1], stepSize):
            for x in range(0, imageCinza.shape[0], stepSize):
                window = imageCinza[x:x + w_width, y:y + w_height]
                #print('Resize window:',window.shape)
                window = cv2.resize(window, (512, 512), interpolation=cv2.INTER_CUBIC)
                window = 1 - (window.reshape((512, 512))/255)

                ENTRADA[0, :, :, 0] = window
                #ENTRADA[0, :, :, 0] = cv2.resize(window (512, 512))
                output_dict = sess.run(tensor_dict,
                                  feed_dict={image_tensor: ENTRADA})
               #print(output_dict['pre#d_a/mul_33'][0])
                ## TRATAMENTO DA SAIDA ##
                SAIDA=output_dict['refined/Sigmoid']
                saida = sk.img_as_ubyte(SAIDA[0, :, :, 0])
                window = sk.img_as_ubyte(window[:])

                #convertendo saida em numpyarray para image
                np_img = np.array(saida)
                #np_img = cv2.resize(np_img, (256, 256))
                img = Image.fromarray(np_img)

                ## CONCATENA BLOCOS ##

                #print('img: ',img.size)
                background.paste(img, (y, x))    
                #cv2.imshow("Output-block", saida)
                #cv2.imwrite("output/image-block-test-0"+str(y)+str(x)+".png", saida)
                background.save('output/outputImageBlocks_2.png')

                #cv2.imwrite('output/CompareImage.png', ImageCompare)
                #cv2.imwrite("output/CONCAT"+str(y)+".png", img_bwa)
                #cv2.waitKey(0)                 # Waits forever for user to press any key
                #cv2.destroyAllWindows()
    print('-'*10+'Concluido'+'-'*10)