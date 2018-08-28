#TESTING git - Successfull
library(devtools)
#install_github('rstudio/reticulate',force=T)
library(reticulate)
library(tensorflow)
#install_tensorflow()
#install_github("rstudio/keras",force=T)
library(keras)
library(imager)
library(readr)
keras::install_keras()



labels <- read_csv("C:/Users/shrey/Downloads/10-monkey-species/monkey_labels.csv")
#View(labels)
names(labels) <- gsub(" ", "_", names(labels))
train_dir = "C:/Users/shrey/Downloads/10-monkey-species/training/"
test_dir = "C:/Users/shrey/Downloads/10-monkey-species/validation/"
work_dir = getwd()
#function to display the images correponding to the respective class
image_show <- function(num_image,label){
  for (i in 1:num_image){
    imgdir <- paste(train_dir,label,sep = "")
    setwd(imgdir)
    imagefile <- sample(list.files(),1)
    img <- load.image(imagefile)
    plot(img)
    print(labels$Common_Name[labels$Label==label])
  }
  setwd(work_dir)
}

image_show(3,"n5")
#Setting up the variables
LR <- as.integer(1e-3)
height<-as.integer(150)
width<-as.integer(150)
channels<-as.integer(3)
seed<-as.integer(786)
batch_size <- as.integer(64)
num_classes <- as.integer(10)
epoch <- as.integer(400)
data_augmentation <- TRUE
num_predictions <- as.integer(20)

#Augmentation
# Training generator
train_datagen <- image_data_generator(
  rescale=as.integer(1%/%255),
  rotation_range=as.integer(40),
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=TRUE,
  fill_mode='nearest')

train_generator = train_datagen$flow_from_directory(train_dir, 
                                                    target_size = list(as.integer(height),as.integer(width)),
                                                    color_mode = "rgb",
                                                    batch_size= as.integer(batch_size),
                                                    seed=as.integer(seed),
                                                    shuffle=TRUE,
                                                    class_mode='categorical')

# Test generator
test_datagen = image_data_generator(rescale=as.integer(1%/%255))
validation_generator = test_datagen$flow_from_directory(test_dir, 
                                                        target_size=list(as.integer(height),as.integer(width)),
                                                        color_mode = "rgb",
                                                        batch_size=as.integer(batch_size),
                                                        seed=as.integer(seed),
                                                        shuffle=FALSE,
                                                        class_mode='categorical')


train_num = train_generator$samples
validation_num = validation_generator$samples
train_num
validation_num

#Building the model
model_keras <- keras_model_sequential()

model_keras %>%
  layer_conv_2d(filters=32, kernel_size=c(3,3),input_shape=c(150,150,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  
  #another 2-D convolution layer
  layer_conv_2d(filter=32 ,kernel_size=c(3,3))  %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  
  layer_conv_2d(filter=64 , kernel_size=c(3,3),padding="same") %>% 
  layer_activation("relu") %>%  
  layer_conv_2d(filter=64,kernel_size=c(3,3) ) %>%  
  layer_activation("relu") %>%  
  layer_max_pooling_2d(pool_size=c(2,2)) %>%  
  layer_dropout(0.25) %>%
  
  #flatten the input 
  layer_flatten() %>%
  
  layer_dense(512) %>%  
  layer_activation("relu") %>%  
  layer_dropout(0.5) %>%  
  
  #output layer-10 classes-10 units
  layer_dense(as.integer(num_classes)) %>% 
  
  #applying softmax nonlinear activation function to the output layer #to calculate cross-entropy#output layer-10 classes-10 units  
  layer_activation("softmax")

model_keras %>%
  compile(optimizer = optimizer_adam(lr = 0.003, decay = 1e-6), loss = "categorical_crossentropy", metrics = c("acc"))

summary(model_keras)

#filepath <- paste(getwd(),"/model.h5f",sep = "")

stepEpoch <- train_num %/% batch_size
Val_num <- validation_num %/% batch_size

history <- model_keras  %>% fit_generator(train_generator,
                              # batch_size = batch_size,
                              steps_per_epoch= as.integer(stepEpoch),
                              epochs=as.integer(epoch),
                              validation_data=validation_generator,
                              validation_steps= as.integer(Val_num))


