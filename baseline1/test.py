# _*_ coding:utf-8 _*_
import sys
caffe_root="/home/ustc/caffe/"
sys.path.append(caffe_root+"python")
import caffe
import numpy as np
import cv2


def main(path_model_def,path_model_wight,path_image_txt):

    model_def=path_model_def
    model_weights=path_model_wight
    net=caffe.Net(model_def,        #defines the structure of the model
                  model_weights,    #contains the trained weights
                  caffe.TEST)       #use test mode(e.g.,don't perform dropout)

    transformer=caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    print("net.blobs[data].data.shape:",net.blobs['data'].data.shape)
    transformer.set_transpose('data',(2,0,1))   #通道变换,(width ,height,channel)->(channel,width,height)
    transformer.set_raw_scale('data',255)       #rescale from [0,1] to [0,255]

    with open(path_image_txt) as image_list:    #该文件的内容一般是：每行表示图像的路径，然后空格，然后标签，也就是说每行都是两列
        while 1:
            list_name=image_list.readline()
            if list_name=='\n' or list_name=='':    #文件读完退出循环
                break
            image=caffe.io.load_image(list_name)    #(width,height,channel),RGB
            #image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            #image1=cv2.imread(list_name)#(BGR)该数据格式为正确格式
            cv2.imshow("input_image",image)
            print("image_shape:",image.shape)
            image=caffe.io.resize_image(image,(35,35,1))

            print("image_resize_shape:",image.shape)
            transformed_image=transformer.preprocess("data",image)
            print("transformed_image.shape:",transformed_image.shape)

            #用转换后的图像代替net.blob中的data
            net.blobs['data'].data[...]=transformed_image
            print("net",net.blobs['data'].data[...].shape)

            out_put=net.forward()
            print(out_put['res'])
            print(out_put['res'].shape)
            out_put['res'] =np.transpose(out_put['res'],(0,2,3,1))   #（1,1,35,35）->(1,35,35,1)
            cv2.imshow("out_put",out_put['res'][0]/255.)
            cv2.waitKey(0)


if __name__=="__main__":
    model_def="./baseline1.prototxt"
    model_weights="./baseline1_qp38.caffemodel"
    image_txt="./image/test.txt"
    main(model_def,model_weights,image_txt)

