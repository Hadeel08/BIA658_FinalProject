# Set data path as per your data file (for example: "c://abc//" )
setwd("/Users/hadoola/Desktop/BIA658/FinalProject/ml-1m")

library(stringr)

# Read the Movies dataset into table */
movies_data = read.csv("./movies.dat",sep= "_", stringsAsFactors = F, header = FALSE)
movies_data = cbind(movies_data,gsub("'s","",movies_data$V3))
#Clean Generes from "'s" like "Children's"
movies_data = cbind(movies_data,str_split_fixed(movies_data$`gsub("'s", "", movies_data$V3)`, "[|]", 6))
names(movies_data) = c("MovieID","Movie_Name","Genres","CleanedGeneres","Genre 1","Genre 2","Genre 3","Genre 4","Genre 5","Genre 6")

# Read the Ratings dataset into table */
ratings_data = read.csv("./ratings.dat",sep= "_", stringsAsFactors = F, header = FALSE)
names(ratings_data) = c("UserID","MovieID","Rating","Timestamp")

# Read the Users dataset into table */
users_data = read.csv("./users.dat",sep= "_", stringsAsFactors = F, header = FALSE)
names(users_data) = c("UserID","Gender","Age","Occupation","Zip-code")

library(reshape2)
#Create ratings matrix. Rows = userId, Columns = movieId
ratingmat <- dcast(ratings_data, UserID~MovieID, value.var = "Rating", na.rm=FALSE)
ratingmat <- as.matrix(ratingmat[,-1]) #remove userIds

library(recommenderlab)
#Convert rating matrix into a recommenderlab sparse matrix
ratingmat <- as(ratingmat, "realRatingMatrix")

#Normalize the data
ratingmat_norm <- normalize(ratingmat)

#Create Recommender Model. "UBCF" stands for User-Based Collaborative Filtering
recommender_model <- Recommender(ratingmat_norm, method = "UBCF", param=list(method="Cosine",nn=30))
recom <- predict(recommender_model, ratingmat[1], n=10) #Obtain top 10 recommendations for 1st user in dataset
recom_list <- as(recom, "list") #convert recommenderlab object to readable list

#Obtain recommendations
recom_result <- matrix(0,10)
for (i in c(1:10)){
  recom_result[i] <- movies_data[as.integer(recom_list[[1]][i]),2]
}

evaluation_scheme <- evaluationScheme(ratingmat, method="cross-validation", k=5, given=3, goodRating=5) #k=5 meaning a 5-fold cross validation. given=3 meaning a Given-3 protocol
evaluation_results <- evaluate(evaluation_scheme, method="UBCF", n=c(1,3,5,10,15,20))
eval_results <- getConfusionMatrix(evaluation_results)[[1]]