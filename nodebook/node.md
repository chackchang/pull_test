const char*image_path=argv[1];
const char*model_path=argv[2];

int  output(const char*image_path,const char*model_path){
    int im_fd=open(image_path);
    int model_fd=open(model_path);
    write(im_fd);
    return model_fd(im_fd);
}