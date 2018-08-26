library(keras)
library(tensorflow)
install_keras()
library(imager)
install_tensorflow()
library(readr)
labels <- read_csv("C:/Users/shrey/Downloads/10-monkey-species/monkey_labels.csv")
View(labels)
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

#Setting up the variables
LR <- 1e-3
height<-150
width<-150
channels<-3
seed<-1337
batch_size <- 64
num_classes <- 10
epochs <- 200
data_augmentation <- TRUE
num_predictions <- 20

