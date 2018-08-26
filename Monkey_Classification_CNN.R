#TESTING git
library(devtools)
#install_github('rstudio/reticulate',force=T)
library(reticulate)
library(tensorflow)
install_tensorflow()
#install_github("rstudio/keras",force=T)
library(keras)
keras::install_keras()
library(imager)
library(readr)


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
}

image_show(3,"n5")
#Setting up the variables
LR <- 1e-3
height<-150L
width<-150L
channels<-3
seed<-1337
batch_size <- 64
num_classes <- 10
epochs <- 200
data_augmentation <- TRUE
num_predictions <- 20

#Augmentation
# Training generator
train_datagen <- image_data_generator(
  rescale=1./255,
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=TRUE,
  fill_mode='nearest')

train_generator = train_datagen$flow_from_directory(train_dir, 
                                                    target_size = list(height,width),
                                                    color_mode = "rgb",
                                                    batch_size= batch_size,
                                                    seed=seed,
                                                    shuffle=TRUE,
                                                    class_mode='categorical')

# Test generator
test_datagen = image_data_generator(rescale=1./255)
validation_generator = test_datagen$flow_from_directory(test_dir, 
                                                        target_size=list(height,width),
                                                        color_mode = "rgb",
                                                        batch_size=batch_size,
                                                        seed=seed,
                                                        shuffle=FALSE,
                                                        class_mode='categorical')


train_num = train_generator$samples
validation_num = validation_generator$samples

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
  layer_dense(num_classes) %>% 
  
  #applying softmax nonlinear activation function to the output layer #to calculate cross-entropy#output layer-10 classes-10 units  
  layer_activation("softmax")

model_keras %>%
  compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = c("binary_accuracy"))

summary(model_keras)

history <- model_keras  %>% fit_generator((train_generator),
                              # batch_size = batch_size,
                              steps_per_epoch= 1097/64,
                              epochs=200,
                              validation_data=(validation_generator),
                              validation_steps= 272/64)
