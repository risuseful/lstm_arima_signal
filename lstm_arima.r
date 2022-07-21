# Ref: http://rwanjohi.rbind.io/2018/04/05/time-series-forecasting-using-lstm-in-r/

rm(list = ls())


library(stringr)
library(keras)
library(tensorflow)

# generate time series

s1 <- arima.sim(model= list(order = c(0, 1, 0)), n=200, mean=1, sd=5)

s2 <- arima.sim(model= list(order = c(0, 1, 0)), n=200, mean=1.5, sd=3)

plot(s1, ylim=c(min(min(s1), 
                    min(s2)),
                max(max(s1), 
                    max(s2))
                )
     )
lines(s2, col="red")




# transform data to stationarity (I'm just using s1)
diffed <- diff(s1, differences=1)
head(diffed)
plot(diffed)


# lagged dataset
lag_transform <- function(x, k= 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}
supervised = lag_transform(diffed, 1)
head(supervised)




## split into train and test sets

N = nrow(supervised)
n = round(N *0.7, digits = 0)
train = supervised[1:n, ]
test  = supervised[(n+1):N,  ]




## scale data
scale_data = function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), scaled_test = as.vector(scaled_test) ,scaler= c(min =min(x), max = max(x))) )
  
}


Scaled = scale_data(train, test, c(-1, 1))

y_train = Scaled$scaled_train[, 2]
x_train = Scaled$scaled_train[, 1]

y_test = Scaled$scaled_test[, 2]
x_test = Scaled$scaled_test[, 1]



## inverse-transform
invert_scaling = function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for( i in 1:t){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}



# Reshape the input to 3-dim
dim(x_train) <- c(length(x_train), 1, 1)

# specify required arguments
X_shape2 = dim(x_train)[2]
X_shape3 = dim(x_train)[3]
batch_size = 1                # must be a common factor of both the train and test samples
units = 1                     # can adjust this, in model tuning phase

#=========================================================================================

model <- keras_model_sequential() 
model%>%
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3), stateful= TRUE)%>%
  layer_dense(units = 1)




model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam( lr= 0.02, decay = 1e-6 ),  
  metrics = c('accuracy')
)




Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=FALSE)
  model %>% reset_states()
}



# Make predictions
L = length(x_test)
scaler = Scaled$scaler # gives min and max of columns
predictions = numeric(L) # generate 0 vectors with length=L

for(i in 1:L){
  X = x_test[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size=batch_size)
  
  # invert scaling
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  
  # invert differencing
  yhat  = yhat + s1[(n+i)]
  
  # store
  predictions[i] <- yhat
}


# plot s1 against prediction
ts1 <- seq(1,length(s1),1)
tp <- seq(length(s1)-length(predictions)+1, length(s1))

plot(ts1, s1, col="blue")
lines(tp, predictions, col="red")
