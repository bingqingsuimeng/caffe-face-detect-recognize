# download repository  
git clone --recursive http://192.168.0.7/root/SenscapeFaceProject.git  
cd SenscapeFaceProject/  
git checkout CameraSurveillance  
git submodule init  
git pull  
git submodule  update  
cd include/dlib-19.4/  
git checkout refs/remotes/origin/face -b face  

# make project  
1、 cd dlib &&mkdir build &&cd build &&cmake .. &&make -j4  
2、 cd ../../../caffe &&mkdir build &&cd build &&cmake .. &&make -j4  
3、 cd ../../../ && mkdir build &&cd build &&cmake .. &&make -j4  
