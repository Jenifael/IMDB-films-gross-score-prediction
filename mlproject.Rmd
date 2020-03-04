---
title: "R for Data Science / ISL Project: The IMDB 5000 Database"
output:
  html_document:
    toc: true
    theme: united
---
CHAPEAU Paul, ELLAOUNI Mehdi, SUBERBIELLE Alexandre, BIZON MONROC Claire


The **IMDB Dataset** is a dataset of approximately 5000 films listed on the online movie database IMDB. It contains various information about the movies themselves (duration, name of director, year, genre...) as well as a score given by the users of the website.

Overall, the dataset has 28 variables:

* **movie_title**
* **duration**	(in minutes)
* **director_name**	Name of the Director of the Movie
* **director_facebook_likes**	Number of likes of the Director on his Facebook Page
* **actor_1_name**	Primary actor starring in the movie
* **actor_1_facebook_likes**	Number of likes of the Actor_1 on his/her Facebook Page
* **actor_2_name**	Other actor starring in the movie
* **actor_2_facebook_likes**	Number of likes of the Actor_2 on his/her Facebook Page
* **actor_3_name**	Other actor starring in the movie
* **actor_3_facebook_likes**	Number of likes of the Actor_3 on his/her Facebook Page
* **num_user_for_reviews**	Number of users who gave a review
* **num_critic_for_reviews**	Number of critical reviews on imdb
* **num_voted_users**	Number of people who voted for the movie
* **cast_total_facebook_likes**	Total number of facebook likes of the entire cast of the movie
* **movie_facebook_likes**	Number of Facebook likes in the movie page
* **plot_keywords	Keywords** describing the movie plot
* **facenumber_in_poster**	Number of the actor who featured in the movie poster
* **color**	Film colorization. ‘Black and White’ or ‘Color’
* **genres**	Film categorization. One film can belong to several genres.
* **title_year**	The year of release.
* ***language**	
* **country**	Country of production
* **content_rating** It the movie advised for a generalaudience ?
* **aspect_ratio** Video Format
* **movie_imdb_link**	IMDB link of the movie
* **gross**	Gross earnings of the movie in its country's currency
* **budget**	Budget of the movie in Dollars
* **imdb_score**	IMDB Score of the movie on IMDB, rated by the users

We will here try to apply machine learning tools to understand the impact of these variables on the IMDB score and gross earnings of a movie. This question will be addresed through 2 regression and 1 classification problems.

```{r message=FALSE, include=FALSE}
library(Amelia)
library(caret)
library(class)
library(corrplot)
library(dplyr)
library(doParallel)
library(glmnet)
library(ggplot2)
library(ggrepel)
library(gridExtra)
library(leaps)
library(pracma)
library(randomForest)
library(ranger)
library(rpart)
library(rpart.plot)
library(tidyverse)
library(visNetwork)
library(sparkline)
library(stringr)
library(pROC)
```




# I) Cleaning the Data

## A) Exploring and understanding our dataset
Read data directly from our website

```{r}
#data=read.csv("movies/movie_metadata.csv", na.strings = c("NA", ""))
data <- read.csv(url('http://suberalex.com/movie_metadata.csv'), na.strings = c("NA", ""))
data =  data %>% distinct() #removing duplicates
```


```{r}
dim(data)
```
Our dataset has 4998 rows and 28 columns. Let's inspect it.

```{r}
summary(data)
```

We have a lot of variables, some of them could turn out to be inexploitable or irrelevant.
What about the missing data ?

```{r}
data %>% summarise_all(funs(mean(is.na(.))))
```


```{r}
missmap(data)
```
We have a lot of missing data on some variables. How should we handle them ? 

How to handle the Facebook likes ? 

```{r}
fb_likes = data %>%  select(contains("likes"))
fb_likes %>% summarise_all(funs(na = mean(is.na(.) | (. == 0), na.rm=T)))
```

```{r}
fb_likes = fb_likes %>% drop_na()
data_cor <- cor(fb_likes)
corrplot(corr = data_cor, method = 'color', addCoef.col="grey")
```

We come up with the folliwing guidelines for data cleaning:

## B) Cleaning

GUIDELINES for data cleaning :

**All the movie titles have a special character A in the end, we need to remove them**

"movie_title"

**Keeping only the TOP N values, packing the remaining lot under "Other"**

"genres" (*note*: one film can have several genres, we'll pack all this in a matrix)

**Filering on values:**
"country": keep only USA, UK and France

**Replacing NA and 0 values with average value**
"duration"
"actor_3_facebook_likes"
"actor_2_facebook_likes"
"actor_1_facebook_likes"

**Other Replacement Policy**
"content_rating": put the NAs in the Not Rated category, merge duplicate ratings


**Processing names**
The procedure for processing names is describes in the next paragraph.

**USELESS variables to remove:**
  "num_critic_for_reviews"
  "plot_keywords"
  "num_user_for_reviews" 
  "movie_imdb_link" 
  "color"
  "aspect_ratio"
  "language" (most are English after filtering for european and US films), 
  "movie_facebook_likes" (almost half unknown)
  "director_facebook_likes": too much missing data
  "cast_total_facebook_likes" (we only the highly correlated keep the 3 actors)

Let's clean our data:

```{r}
data$movie_title <- gsub("Â", "", as.character(factor(data$movie_title)))

c_ratings = c("Approved"="G", "GP"="PG", "Unrated"="NC-17", "Not Rated"="NC-17", "Passed"="G", "X"="NC-17", "M"="NC-17")
dim(data)
data_year = data %>% filter(title_year!=2015)
dim(data_year)
data_country = data_year %>% filter(grepl("(UK|USA|France)", country))
dim(data_country)
data_cleaned = data_country %>% drop_na("num_voted_users", "facenumber_in_poster", "budget", "imdb_score", "title_year", "gross", "director_name", "actor_1_name")
dim(data_cleaned)
data_cleaned=data_cleaned %>% 
  mutate(content_rating=factor(recode_factor(content_rating, !!!c_ratings))) %>%
  #if NA, we consider it for General Audience
  mutate(content_rating=replace_na(content_rating, "G" )) %>%
  #change factors into characrers, will be easier for further processing %>%
  mutate(actor_1_name=as.character(actor_1_name), actor_2_name=as.character(actor_2_name),
         actor_3_name=as.character(actor_3_name), director_name=as.character(director_name))
```

```{r}
#Drop useless variables
data_cleaned = data_cleaned %>% select(-c(num_critic_for_reviews, plot_keywords, num_user_for_reviews, movie_imdb_link, color, aspect_ratio, language, movie_facebook_likes, director_facebook_likes, cast_total_facebook_likes))
dim(data_cleaned)
```

*Processing names*
We will replace director names with a variable coding for the average IMDB score of their previous movies. 
We cannot avoid NA values: every director has at least ONE unkown value corresponding to the first of his film appearing in the database. For one-time director, we will therefore never have any meaningful score.
Moreother, some directors did more than one film in their first year, and our method doesn't give any value to all theses films either.
So the total number of unknown values should be:
number of distinct directors **+** ((number of films did the first year - 1) by director)

```{r}
first_y = data_cleaned %>% 
  group_by(director_name, title_year) %>% 
  summarise(count=n()) %>% ungroup() %>%
  group_by(director_name) %>% 
  top_n(-1, title_year) %>% ungroup() %>%
  mutate(count = count -1) %>%
  summarise(count=sum(count))
(first_y + n_distinct(data_cleaned$director_name)) / nrow(data_cleaned)
```
 This is a lot, but still retains more information that the other option which is to code the top N directors as dummies, and allows us to have a continuous variable.
  We will replace these NA values with the average score by director.
  We implement the same procedure for the main actor of each film. 

```{r}
get_director_score = function(df, name, year) {
  
  avg = df %>% ungroup() %>% group_by(director_name) %>%
    summarise(imdb=mean(imdb_score)) %>%
    summarise(mean(imdb, na.rm=T)) %>%
    as.numeric()
  
  filtered = df %>% ungroup() %>%
    filter(director_name == name, title_year < year) 
  
  filtered %>%
    summarise(mean_score=ifelse(nrow(filtered) > 0, mean(imdb_score), avg)) %>%
    as.numeric()
} 

get_director_score(data_cleaned, "Martin Scorsese", 1989)
```

```{r}
get_actor_score = function(df, name, year) {
  
  avg = df %>% ungroup() %>% group_by(actor_1_name) %>%
    summarise(imdb=mean(imdb_score)) %>%
    summarise(mean(imdb, na.rm=T)) %>%
    as.numeric()
  
  filtered = df %>% ungroup() %>%
    filter(actor_1_name == name | actor_2_name == name | actor_3_name == name, title_year < year) 
  
  filtered %>%
    summarise(mean_score=ifelse(nrow(filtered) > 0, mean(imdb_score), avg)) %>%
    as.numeric()
} 

get_actor_score(data_cleaned, "Brad Pitt", 1995)
```

```{r}
data_cleaned=data_cleaned %>% rowwise() %>%
  mutate(director_score = get_director_score(data_cleaned, director_name, title_year),
         main_actor_score = get_actor_score(data_cleaned, actor_1_name, title_year)) %>%
  ungroup() %>%
  select(-director_name, -actor_1_name, -actor_2_name, -actor_3_name)
```

Replace NA and 0s with average values:

```{r}
data_cleaned = data_cleaned %>%
  mutate(
    # is NA if 0
    duration = na_if(duration, 0),
    actor_1_facebook_likes = na_if(actor_1_facebook_likes, 0),
    actor_2_facebook_likes = na_if(actor_2_facebook_likes, 0),
    actor_3_facebook_likes = na_if(actor_3_facebook_likes, 0)
  ) %>% 
  mutate(
    duration = replace_na(duration, mean(duration, na.rm=T)),
    actor_1_facebook_likes = replace_na(actor_1_facebook_likes, mean(actor_1_facebook_likes, na.rm=T)),
    actor_2_facebook_likes = replace_na(actor_2_facebook_likes, mean(actor_2_facebook_likes, na.rm=T)),
    actor_3_facebook_likes = replace_na(actor_3_facebook_likes, mean(actor_3_facebook_likes, na.rm=T))
  )
```


```{r}
missmap(data_cleaned)
```

```{r}
data_cleaned %>% summarise_all(funs(mean(is.na(.))))
```

No more NAs ! 

```{r}
dim(data_cleaned)
```

```{r}
nrow(data_cleaned) / nrow(data) * 100
```
We keep ~67% of our data !

Let's visualize the impact of the cleaning procedure:

```{r}
comparing = data %>% select(imdb_score, gross)  %>% filter(!is.na(gross)) %>%
            bind_rows(data_cleaned %>% select(imdb_score, gross), .id="dataset") 
ggplot(comparing)+aes(y=imdb_score)+geom_boxplot(color="purple")+facet_wrap(~dataset)
```

```{r}
summary(data$imdb_score);summary(data_cleaned$imdb_score)
print("/n")
summary(data$gross);summary(data_cleaned$gross)
```

Looks like we preserved a good representation of our data !

```{r}
head(data);head(data_cleaned)
```


## C) MONETARY VALUES : conversion to dollars and taking in count inflation

We have 2 values to convert in dollar/correct from inflation: gross and budget

We will base our study on 2019 US$.

```{r}
inflationdb=read.csv(url('http://suberalex.com/inflation_US.csv'), sep="\t") # Inflation rates for US $ from 1914 to 2018
inflationdb=inflationdb %>% select(Year,Annual)
year2019=2019
EURUSrate=1.15 #as of 1st january 2019, 1 EUR is equal to 1.15 USD
POUNDUSrate=1.25 #as of 1st january 2019, 1 GBP is equal to 1.25 USD
```


```{r}
actualizationf= function (value,year,targetyear,db) {
n=targetyear-year
for (i in 1:n) {
  value=value*(1+db[n+1-i,2]/100) #Actualization function
}
value
}
```


```{r}
#Monetary conversion
data_cleaned_a=data_cleaned %>% mutate(budget=ifelse(country=="USA",budget,ifelse(country=="UK",budget*POUNDUSrate,budget*EURUSrate)),gross=ifelse(country=="USA",gross,ifelse(country=="UK",gross*POUNDUSrate,gross*EURUSrate)))
#Actualization
data_cleaned_a=data_cleaned %>%  rowwise() %>% mutate(budget=sapply(budget,actualizationf,year=title_year,targetyear=2019,db=inflationdb)) %>% mutate(gross=sapply(gross,actualizationf,year=title_year,targetyear=2019,db=inflationdb)) %>%
  ungroup()
```

Adding a new category "ROI" and "profit" that will be useful for later

```{r}
data_cleaned_a=data_cleaned_a %>% mutate(ROI=gross/budget,profit=gross-budget)

data_cleaned=data_cleaned %>% mutate(ROI=gross/budget)

ggplot(data_cleaned)+aes(y=gross-budget,x=title_year)+geom_point()+geom_smooth(method="lm",se=F)
ggplot(data_cleaned_a)+aes(y=gross-budget,x=title_year)+geom_point()+geom_smooth(method="lm",se=F)

```
We have now a better view on the different monetary values in our dataset.

Let's check with the highest gross movie on our dataset: 
```{r}
data_cleaned_a$movie_title[which.max(data_cleaned_a$gross)]
```

It seems to be a great estimation of the adjusted profit of our dataset."Gone with the Wind" is indeed the top 1 film worldwide when we adjust for inflation. 

Source: https://www.cnbc.com/2019/07/22/top-10-films-at-the-box-office-when-adjusted-for-inflation.html


## D) Processing genres

We have a lot of genres and we need to create dummy variables according to those genres. However, we will focus on the top 10 most represented genres so that we won't have too much dummy variables.


```{r}
TOP_N=10
genres = data_cleaned_a %>% group_by(genres)  %>% summarise(count=n()) %>%
  separate(col=genres, sep="\\|", into=c("GenreA", "GenreB", "GenreC")) %>%
  gather(key=genre_num, value=genre, -count) %>% 
  group_by(genre) %>% summarise(count=sum(count)) %>%
  filter(!is.na(genre)) %>% ungroup() %>% 
  top_n(TOP_N, count)
genres
```

```{r}
movies_genre = data_cleaned_a %>% select(movie_title, budget,imdb_score,title_year, all_genres=genres)
for (genre in genres$genre) {
  movies_genre = movies_genre %>% mutate(!!genre := as.numeric(grepl(genre, all_genres))) 
}
#write.csv(movies_genre, "movies_genre.csv")
```

```{r}
movies_genre
```



# II) Data Visualization

## A) IMDB Score

### > Global view

```{r}
ggplot(data_cleaned_a)+aes(x=imdb_score,y = ..density..)+ geom_histogram(bins=30,fill="Purple")+geom_density(size=1.5,color="yellow")
  scale_x_continuous(name = "IMDB Score",
                     breaks = seq(0,10),
                     limits=c(1, 10))
```

### > by country
```{r}
ggplot(data_cleaned_a)+aes(x=country,y=imdb_score,fill=country)+
  geom_boxplot()
```

### > by content_rating
```{r}
ggplot(data_cleaned_a)+aes(x=content_rating,y=imdb_score,fill=content_rating)+
  geom_boxplot()+coord_flip()
```

### > facebook like vs imdb score
```{r}
data_cleaned_a%>%mutate(totalfacooklikes=actor_1_facebook_likes+actor_2_facebook_likes+actor_3_facebook_likes)%>%
  ggplot()+
  aes(x=log(totalfacooklikes),y=imdb_score)+
  geom_point(size=0.5)+ 
  geom_smooth(method="lm",se=F,color="red")+
  facet_wrap(~country,ncol = 3)
```

We can see that Facebook Like have a positive impact on Imdb_score. The more famous the actor are, the better the IMDB score should be.

### > By genres
```{r}
movies_genre%>%gather(key="genre",value="value",-movie_title,-budget,-imdb_score,-title_year,-all_genres)%>%group_by(genre)%>%summarise(mean_genre=mean(value))%>%arrange(-mean_genre)
```
TOP6
Drama
Comedy
Thriller
Action
Romance
Adventure


```{r}
movies_genre%>%select(-movie_title,-budget,-all_genres)%>%
  filter(title_year>1925)%>%
  gather(key="genre",value="value",-imdb_score,-title_year)%>%
  filter(grepl("(Drama|Comedy|Thriller|Action|Romance|Adventure)",genre))%>%
  mutate(imdb_score_genre=imdb_score*value)%>%
  group_by(genre,title_year)%>%
  summarise(imdb_score_year=mean(replace(imdb_score_genre, imdb_score_genre==0, NA), na.rm = T))%>%mutate(missing=interp1(title_year,imdb_score_year,title_year,"linear"))%>%
  ggplot()+
  aes(x=title_year,y=imdb_score_year)+
  geom_line(na.rm=T,color="blue")+
  geom_point(na.rm=T,color="blue",size=0.5)+
  geom_smooth(method="lm",se=F,color="red",na.rm=T)+
  geom_line(aes(x=title_year,y=missing),na.rm=T,color="blue",linetype="dashed",size=0.5)+
  xlab("Year of release")+
  ylab("Imdb Score")+
  facet_wrap(~genre,ncol=3,nrow=2)
```
  
We can see that for each kind of movie, there is a negative impact of the year for the IMDB predict score. It might be because when someone is reviewing an old movie, he's less likely to give a bad grade because reviewers of old movies are most likely to be already fans. 

```{r}
movies_genre%>%select(-movie_title,-budget,-all_genres)%>%
  gather(key="genre",value="value",-imdb_score,-title_year)%>%
  filter(title_year>1980&title_year<2015)%>%
  filter(grepl("(Drama|Comedy|Thriller|Action|Romance|Adventure)",genre))%>%
  group_by(genre,title_year)%>%
  summarise(numberofmovie=sum(value))%>%
  ggplot()+
  aes(x=title_year,y=numberofmovie,fill=genre)+
  xlab("Year of release")+
  ylab("Proportion of genre")+
  ggtitle("Repartition of year trough time")+
  geom_bar(stat = "identity",position="fill")
```

### > By duration

```{r}
p1=data_cleaned_a%>% select(imdb_score, duration)%>%
  ggplot()+
  aes(x=log(duration),y=imdb_score)+
  geom_point(size=0.5)+ 
  geom_smooth(method="lm",se=F,color="red")+
  geom_text(aes(6,7.5,label=cor(duration, imdb_score)),color="red")+
  xlab("Duration - log")+
  ylab("Imdb Score")
p2=data_cleaned_a%>% select(profit, duration)%>%
  ggplot()+
  aes(x=log(duration),y=profit)+
  geom_point(size=0.5)+ 
  geom_smooth(method="lm",se=F,color="red")+
  geom_text(aes(6,6,label=cor(duration, profit)),color="red")+
  xlab("Duration - log")+
  ylab("Profit")
grid.arrange(p1, p2, nrow = 1)
```

```{r}
p3<-ggplot(data_cleaned)+
  aes(x=director_score,y=imdb_score)+
  geom_point(size=0.5)+geom_smooth(method="lm",se=F,color="red")+
  xlab("Director Score Previous Score")+
  ylab("Imdb Score")
p4<-ggplot(data_cleaned)+
  aes(x=main_actor_score,y=imdb_score)+
  geom_point(size=0.5)+geom_smooth(method="lm",se=F,color="red")+
  xlab("Main Actor Previous Score")+
  ylab("Imdb Score")
grid.arrange(p3, p4, nrow = 1)
```

## B) GROSS vs IMDB Score

```{r}
data_cleaned_a%>%mutate(profit=(gross-budget))%>%top_n(profit,100) %>% ggplot()+
  aes(x=imdb_score,y=gross,size=profit,color=country)+
  geom_point()+
  geom_hline(aes(yintercept = 3.7e+08)) + 
  geom_vline(aes(xintercept = 6.5)) 
```


```{r}
ggplot(data_cleaned_a)+aes(y=gross,x=imdb_score)+geom_point()+geom_smooth(method="lm",se=F)
```

## C)Top 20 films based on ROI
```{r}

data_cleaned_a %>%
  filter(budget > 100000) %>%
  arrange(desc(gross-budget)) %>%
  top_n(10, gross-budget) %>%
  ggplot(aes(x=budget, y=gross-budget)) +
  geom_point() +
  geom_smooth() + 
  geom_text_repel(aes(label=movie_title)) +
  labs(x = "Budget $million", y = "Profit", title = "Top 10 Profitable Movies with a budget > 100k$") +
  theme(plot.title = element_text(hjust = 0.5))
```



# III) A REGRESSION PROBLEM: can we predict a film's IMDB score ? -  Linear Model and subset selection

First task: predicting the film's IMDB score before its debut: therefore for this task we will not use 
- the gross
- the number of users who votes on IMDB 

We select only the variables that will be interesting for predicting imdb score. 


```{r}
data.imdb= data_cleaned_a %>% 
  left_join(movies_genre, by=c("movie_title", "budget", "title_year", "imdb_score")) %>%
  select(-all_genres, -genres, -gross, -movie_title, -num_voted_users,-profit,-director_score,-main_actor_score) 
```


 We are going to perform a *subset selection* to select the best linear models we can have with our data with a stepwise fit.

We split the data into 2 a training dataset of 75% of the data and a test dataset of 25% of the dataset

```{r}
set.seed(1234)
n=nrow(data.imdb)
samp <- sample(n)
i.train <- samp[1:round(0.75*n)]
i.test <- samp[-c(1:round(0.75*n))]
train <- data.imdb[i.train,]
test <- data.imdb[i.test,]
```

```{r}
linearmodel.imdb=lm(imdb_score~.,data=train)
```

We observe that only duration, title_year, and documentary,drama,horror,thriller genres are significant
Our R square is quite low, maybe a linear model is not adapted to the model. 

```{r}
summary(linearmodel.imdb)
```


```{r}
#We select all the statistically significant variables
selected=summary(linearmodel.imdb)$coeff[-1,4] < 0.05
# select sig. variables
relevant <- names(selected)[selected == TRUE] 
relevant[5]="content_rating"
relevant=relevant[-c(4,6)]

# formula with only sig variables
formula <- as.formula(paste("imdb_score ~",paste(relevant, collapse= "+")))
linearmodel.imdb.rel=lm(formula=formula,data=train)
#summary(linearmodel.imdb.rel)

#Select columns on train

train2=train %>% select(imdb_score,relevant)
test2=test %>% select(imdb_score,relevant)
mod.sel <- regsubsets(imdb_score~.,data=train2)
summary(mod.sel)

```

```{r}
plot(mod.sel,scale="bic")
plot(mod.sel,scale="Cp")
```

Both bic and cp models looks the same


```{r}
mod.biccp <- lm(imdb_score~duration+content_rating+Drama+Horror,data=train)

#Compute error
prev <- data.frame(Y=test2$imdb_score,lin=predict(linearmodel.imdb.rel,newdata=test2),BIC=predict(mod.biccp,newdata=test2))
prev %>% summarize(Err_lin=mean((Y-lin)^2),Err_BIC=mean((Y-BIC)^2))
```

We have an error on the model selected that is higher than the one of the linear model. Also, the error rates are quite high on both models. The variable selection method seems then not to suit our approach of this dataset. Let's try to use regularized methods. We saw the imdb_score, we are now going to perform go further with Lasso and Ridge methods for the gross on our imdb data set.


# IV) A REGRESSION PROBLEM: can we predict a film's gross box office ? - regularized methods


First task: predicting the film's gross before its debut: therefore for this task we will not use 
- the IMDB score 
- the number of users who votes on IMDB 
We also remove the movie title, and use our dummy variables to code for genre.
Since the average gross box office is around 48M, we change this variable unit to be expressed in Million USD.

We rename columns with the hyphen "-" in their names to avoid subsequent errors.
We create dummies for name variables.

We are likely to have a lot of dummies with underrepresented values and should be careful, especially since we intend to use successive cross validations. Let's inspect our dummies:

```{r}
movies_genre %>% select(Action:"Thriller")  %>%
 apply(2, sum)
```

```{r}
data.reg = data_cleaned_a %>% 
  left_join(movies_genre, by=c("movie_title", "budget", "title_year", "imdb_score")) %>%
  select(-all_genres, -genres, -imdb_score, -movie_title, -num_voted_users, -profit, -ROI) %>%
  mutate(gross=gross/1e6) %>%
  rename(Sci_fi="Sci-Fi")
#We compute the dummies directly to avoid being annoyed by the knn reg function
data.reg = data.frame(model.matrix(lm(gross~., data=data.reg)))[,-1] %>% 
           mutate(gross=data.reg$gross)
```


We will compare the performances of several learning algorithms on this task within an Empirical Risk Minimization Framework.
**metric**: The risk of a model will be estimated with the Root Mean Squard Error.

**Evaluation Procedure**
We are going to randomly split our data in a training, validation and test dataset. 
Every candidate model's hyperparameters will be chosen with a cross validation done on the training dataset.
The models will then be compared with a cross validation on the validation dataset.
We will finally evaluate our best model's performance on the test dataset.
To increase the accuracy of the evaluation, we can repeat this procedure several times.

Finally, we will look at an example application of this model.
Since we have more than 3000 individuals, we can split our data in a  50/40/10 ratio for training / validation / testing respectively. 


For the k-Nearest Neighbours: the number of neighbours.
For the regression trees: the depth of the tree.
For the random forest: the depth of every "simple model" tree. 


Let's define 2 functions for our metrics:
```{r}
RMSE = function(yobs, yhat) sqrt(mean((yhat - yobs)^2, na.rm=T))
MAPE = function(yobs, yhat) mean(abs((yhat - yobs)/yobs), na.rm=T)
```

And initialize the splitting: 

```{r}
N_FOLD=10
set.seed(1235)
train.p = 0.5
val.p = 0.4
n = nrow(data.reg)
train = sample(n, round(train.p * n))
val = sample((1:n)[-train], round(val.p * n))
reg.train = data.reg %>% slice(train)
reg.val = data.reg %>% slice(val)
reg.test = data.reg %>% slice(-c(train, val))
n.train=nrow(reg.train)
n.val=nrow(reg.val)
```

Let's start with a simple linear model with a stepwise selection algorithm.
We use a backward selection process with a BIC criteria, and compare it to a regression k-Nearest Neighbors algorithm, following the procedure explained earlier.

```{r}
#Quick check: since we have a lot of dummy variables with under-represented values:
qr(reg.train)$rank == ncol(reg.train)
```


Let's initialize our parallel coputing tools for faster processing:

```{r}
#For faster computations
cl <- detectCores() %>% -1 %>% makeCluster
cl
registerDoParallel(cl)
```

We start with simple models: 
 * one full linear model
 * a linear model selected with a stepwise procedure from the full linear model
 * a kNN regression model
 
 We will keep our tree at random:

```{r}
#Selecting best linear model with a backward stepwise procedure on the trainoing data
full.linear.model = lm(gross~., data=reg.train)
model.step = step(full.linear.model, direction="backward", k=log(n.train), trace=0)

set.seed(12345)
#Initiating the cross validation method
ctrl.cv = trainControl(method="cv", number=N_FOLD)

#Selecting the best parameters with the cross validation
#Selecting the parameter "k" for kNN 
grid.k = data.frame(k=seq(1,70,by=1))
sel.k = train(gross~., data=reg.train, method="knn", trControl=ctrl.cv, tuneGrid=grid.k)
plot(sel.k)
best.k = which.min(sel.k$results$RMSE)
print(paste("Best k for k-NN: ", best.k))

#Selecting the best parameter "cp" for a regression tree wth 10-fold cross validations
tree = rpart(gross~., data=reg.train, cp=0.00001, xval=N_FOLD, minsplit=2)
best.cp = tree$cptable %>% as.data.frame() %>% 
  filter(xerror==min(xerror)) %>% 
  select(CP) %>%
  as.numeric()
print(paste("Best cp for regression tree: ", best.cp))

#Comparing algorithms
set.seed(1234)
folds = createFolds(1:n.val, k=N_FOLD)

result = foreach(fold=iter(folds), .packages=c("tidyverse", "caret", "rpart", "rpart.plot", "ranger"), .combine=rbind) %dopar% {
  
  val.train = reg.val %>% slice(-fold)
  val.test = reg.val %>% slice(fold)
  
  eval.df = data.frame(yhat.knn=NA, yhat.stepwise=NA, yhat.tree=NA, yhat.lm=NA,          
                       yobs=val.test$gross)
  
  #Predicting with a simple full linear model
  lm =lm(gross~., data=val.train)
  eval.df$yhat.lm = predict(lm, newdata=val.test)
  
  #Predicting with stepwise selected linear model
  stepwise.lm = lm(model.step$call, data=val.train)
  eval.df$yhat.stepwise = predict(stepwise.lm, newdata=val.test)
  
  #Predicting with KNN, selected K
  eval.df$yhat.knn = knnregTrain(val.train, val.test, val.train$gross, k=best.k)
  
  #Predicting with a regression tree
  tree = rpart(gross~., data=val.train, cp=best.cp)
  best.tree = prune(tree, cp=best.cp)
  eval.df$yhat.tree = predict(best.tree, newdata=val.test)
  
  eval.df %>% summarise_all(funs(MAPE(yobs, .), RMSE(yobs, .))) %>%
    select(contains("yhat")) %>%
    gather(id, value) %>%
    separate(id, c("model", "metric"), sep="_") %>%
    spread(key=metric, value=value)
  
}

tab = result %>% group_by(model) %>% 
  summarise(RMSE_sd=sd(RMSE), MAPE=mean(MAPE), RMSE=mean(RMSE))
tab
```

There seems to be a lot of variance in our results.

Let's look at our last best tree:

```{r}
tree = rpart(gross~., data=reg.val[folds[[10]],], cp=best.cp, model=T)
rpart.plot(prune(tree, cp=best.cp))
```

All our errors quite large, indicating that maybe the variables present in the data are not sufficient to correctly predict the gross of a movie. But can we try to decrease the variance of our prediction ? 
For this purpose, we evalute on the same task:
* a random forest algorithm
* a Ridge regression
* a Lasso regression
To keep our results comparable, we evaluate them on the same fold partition. 

```{r}

#Selecting the best parameter "mtry" for a random forest regression with 10-fold cross validations
grid.mtry = data.frame(mtry=seq(1, 30, by=5))
sel.mtry = train(gross~., data=reg.train, method="rf", trControl=ctrl.cv, tuneGrid=grid.mtry)
plot(sel.mtry)
best.mtry = sel.mtry$results$mtry[which.min(sel.mtry$results$RMSE)]
print(paste("Best mtry for Random Forest: ", best.mtry))

set.seed(12)
#Selecting the best parameters for Ridge and Lasso Regressions
train.X = model.matrix(gross~., data=reg.train)[,-1]

ridge.k = cv.glmnet(train.X, reg.train$gross, alpha=0, 
                    lambda=exp(seq(-4, 10,length=100)),
                    nfolds=N_FOLD)
plot(ridge.k)
best.ridge.lambda = ridge.k$lambda.min
print(paste("Best lambda for Ridge Regression: ", best.ridge.lambda))

lasso.k = cv.glmnet(train.X, reg.train$gross, alpha=1, 
                    lambda=exp(seq(-5, 8,length=100)),
                    nfolds=N_FOLD)
plot(lasso.k)
best.lasso.lambda = lasso.k$lambda.min
print(paste("Best lambda for Lasso Regression: ", best.lasso.lambda))

result2 = foreach(fold=iter(folds), .packages=c("tidyverse", "ranger"), .combine=rbind) %dopar% {
  library("glmnet")
  val.train = reg.val %>% slice(-fold)
  val.test = reg.val %>% slice(fold)
  train.X = model.matrix(gross~., data=val.train)[,-1]
  test.X = model.matrix(gross~., data=val.test)[,-1]
  
  eval.df = data.frame(yhat.rf=NA, yhat.ridge=NA, yhat.lasso=NA, yobs=val.test$gross)
  
  #Predicting with Random Forest
  rf = ranger(gross~.,data=val.train,mtry=best.mtry)
  eval.df$yhat.rf = predict(rf, data=val.test)[1]$predictions
  
  #¨Predictions with penalized regressions
  ridge=glmnet(train.X, val.train$gross, alpha=0, lambda=best.ridge.lambda)
  eval.df$yhat.ridge = predict(ridge, newx=test.X)
  
  lasso=glmnet(train.X, val.train$gross, alpha=1, lambda=best.lasso.lambda)
  eval.df$yhat.lasso = predict(lasso, newx=test.X)
  
  eval.df %>% summarise_all(funs(MAPE(yobs, .), RMSE(yobs, .))) %>%
    select(contains("yhat")) %>%
    gather(id, value) %>%
    separate(id, c("model", "metric"), sep="_") %>%
    spread(key=metric, value=value)
  
}

tab2 = result2 %>% group_by(model) %>% 
  summarise(RMSE_sd=sd(RMSE), MAPE=mean(MAPE), RMSE=mean(RMSE))
tab2
```

Inspecting our last LASSO model
```{r}
train.X = model.matrix(gross~., data=reg.val[folds[[10]],])[,-1]
lasso=glmnet(train.X, reg.val$gross[folds[[10]]], alpha=1, lambda=best.lasso.lambda)
lasso$beta
```

Our final results
```{r}
rbind(tab, tab2) %>% arrange(RMSE)
```


Not much variance reduction in penalized regressions ...
We have two good candidates:
LASSO: very small MAPE, better than average RMSE
Random Forest: very small RMSE, small variance.


Let's chose the RANDOM FOREST model.
Let's train our chosen model and evaluate its performance on the test dataset. 


```{r}
#Predicting with Random Forest
set.seed(123456)
reg.final = data.reg %>% slice(c(train, val)) 
model = ranger(gross~.,data=reg.final,mtry=best.mtry)
preds = predict(model, data=reg.test)[1]$predictions
RMSE(reg.test$gross, preds)
MAPE(reg.test$gross, preds)
```
**We have a final MAPE score of 33%.**
Our test score is better than our training score ?? 
Maybe this is a strong indication that we have a lot of variance in our dataset. 
Ideally we should repeat the whole process several times to have a better estimation of our test performance, but this requires a lot of time / computation, we will not do it here.

A concrete example: **The Avengers**:
Let's take a film from our test data.
```{r}
t(data_cleaned_a %>% filter(gross==reg.test$gross[1]*1e6))
```

```{r}
preds[1]
```

We predicted a gross of 366M. 
Th film did a gross of 695M.

# V) A CLASSIFICATION PROBLEM
*******

    Guidelines for creating classes of movies:

We want to classify the movies according to 4 groups:

- Movies with good scores and good profits: These are the excellent movies with good earnings that reflect the critical and public acclaim they had (class A)

- Movies with good scores but bad profits: These are still good movies because they were well received by critics and to some extent the general population but didn't do as well on theaters because of other reasons (class B)

- Movies with bad scores but good profits: ok movies that are still popular even if they are of poor quality (class D)

- Movies with bad scores and bad profits: these are the worst because they failed on theaters and scored bad (class F)

**
For the profits, rather than using the gross, because it penalizes small budget and independant films, we are using the Return On Investment to create a new variable: ROI=GROSS/BUDGET
**
We'll take a look at the means and medians of the variables SCORE and ROI to determine the threshold between good and bad.


```{r}
data.classification = data_cleaned_a %>% mutate(ROI=gross/budget)

summary(data.classification$ROI)

```

**
For the ROI we will take the median 1.143 as the threshold because the mean is affected by outliers

```{r}
summary(data.classification$imdb_score)

```

**
For the score we set the threshold for 6.5 as we observe the median and mean are about the same.

```{r}
data.classification=data.classification %>% mutate(score_class=ifelse(imdb_score<6.5 & ROI<1.143,"F",ifelse(imdb_score<6.5 & ROI>=1.143,"D",ifelse(imdb_score>=6.5 & ROI<1.143,"B","A"))))
data.classification$score_class=as.factor(data.classification$score_class)

ggplot(data.classification)+aes(y=log(actor_1_facebook_likes),x=log(budget),color=score_class)+geom_point()+theme_classic()
ggplot(data.classification)+aes(y=log(gross),x=log(budget),color=score_class)+geom_point()+theme_classic()
ggplot(data.classification)+aes(y=log(gross),x=log(budget),color=score_class)+geom_point()+theme_classic()
ggplot(data.classification)+aes(x=imdb_score,y=log(gross),color=score_class)+geom_point()+theme_classic()
```

**
For the classification problem, we will not use the following variables:
- the ROI, the IMDB score, the gross and the budget because they were used to create the classes.
- the actor and director scores because they are lineary dependent on the imdb_score
- the profit because it depends on the budget and gross.
- the number of users who votes on IMDB because we want to predict the class of the movie when it comes out.
- the movie title because it's irrelevant.

We will also use the same dummy variables to code for genre.

```{r}
data.classification = data.classification %>% 
  left_join(movies_genre, by=c("movie_title", "budget", "title_year", "imdb_score")) %>%
  dplyr::select(-all_genres, -genres, -imdb_score, -movie_title, -num_voted_users, -profit, -budget, -gross, -ROI, -title_year, -director_score, -main_actor_score) %>%
  rename(Sci_fi="Sci-Fi")
data.classification=droplevels(data.classification)

#We will create dummy variables for the columns content_rating and country for the knn classification
library(fastDummies)
data.classification = dummy_cols(data.classification, select_columns = "content_rating")
data.classification = dummy_cols(data.classification, select_columns = "country")
data.classification = data.classification %>% 
  dplyr::select(-content_rating)
data.classification = data.classification %>% 
  dplyr::select(-country)
summary(data.classification)
```

    Classifying the data

We want to classify the data into the 4 previously defined classes. The names of the classes were chosen to mimic a grading system:
- A for excellent movies.
- B for good movies.
- D for average movies.
- F for bad ones.

We are going to compare four classification methods using RMSE as a risk metric:
-k-Nearest Neighbours: the number of neighbours.
-Random forest

The evaluation protocol will be the same as for the regression algorithms, using a training set, a validation and a test set.
The hyperparameters will be chosen using a 10-fold cross validation using the Accuracy

Splitting the dataset:
```{r}
train.classification = data.classification %>% slice(train)
val.classification = data.classification %>% slice(val)
test.classification = data.classification %>% slice(-c(train, val))
n.train=nrow(train.classification)
n.val=nrow(val.classification)
```


**KNN Classification**

selecting best k using 10-fold cross-validation
```{r}
ctrl1=trainControl(method="cv",number=10)
grid.k=data.frame(k=seq(1,50,by=1))
sel.k=train(score_class~.,data=train.classification,method="knn",trControl=ctrl1,tuneGrid=grid.k)
plot(sel.k)
#The best K will be selected with the Accuracy
best.k = which.max(sel.k$results$Accuracy)


print(paste("Best k for k-NN: ", best.k))
```



**Random forest**

selecting best mtry for a random forest using 10-fold cross validations
```{r}
grid.mtry = data.frame(mtry=seq(1, 20, by=5))
sel.mtry = train(score_class~., data=train.classification, trControl=ctrl.cv, tuneGrid=grid.mtry)
plot(sel.mtry)
#The best mtry will be selected with the Accuracy
sel.mtry
best.mtry=6
print(paste("Best mtry for Random Forest: ", best.mtry))
```

The most relevent variable is the duration
```{r}
varImp(sel.mtry)

```

The most significant variable is the duration. It is quite surprising because you rarely think about the duration of the movie to determine its sucess/quality.

The cast facebook likes being the second most important variables is more relevant since it gives us an idea about the notoriety of the movie


**Comparing the models**
We will compare the models using Area under ROC curve and the prediction Error

```{r}
levels(val.classification$score_class) = c(1,2,3,4)
colnames(val.classification)=make.names(colnames(val.classification))
result = foreach(fold=iter(folds), .packages=c("tidyverse", "class", "rpart", "rpart.plot", "ranger", "pROC"), .combine=rbind) %dopar% {
  
  val.train = val.classification %>% slice(-fold)
  val.test = val.classification %>% slice(fold)
  
  eval.df = data.frame(yhat.knn=NA, yhat.rf=NA,
                     yobs=val.test$score_class)
  
  #Predicting using KNN with the selected K
  eval.df$yhat.knn = knn(val.train, val.test, val.train$score_class, k=best.k)
  
  #Predicting using Random Forest with the selected mtry
  rf = ranger(score_class~.,data=val.train,mtry=best.mtry)
  eval.df$yhat.rf = predict(rf, data=val.test)[1]$predictions
  
  eval.df %>% summarise_all(funs(mean((yobs!=.)))) %>%
    select(contains("yhat")) %>%
    gather(id, value) %>%
    separate(id, c("model", "metric"), sep="_") %>%
    spread(key=metric, value=value)
  
}

result %>% group_by(model) %>% 
 summarise_all(mean)
```


Classification conclusion:

- The best algorithm for classifying the data is the random forest 

BUT still the accuracy is just of 43%.


# VI) Conclusion 

Despite our efforts on processing our dataset, we noticed that it is really challenging to determine and perform a regression analysis on it. The different results on both classification and regression are not excellent.

This shows that the information provided in the dataset isn't enough to properly classify a movie nor predict its score or gross.
The information we get from the results is quite poor, as the most significant variable to classify a movie is by its duration. 



