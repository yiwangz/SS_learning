library(ggplot2)
library(reshape2)
library(cowplot)

########################################################################################
## matrices to save the results ########################################################
lam_mat <- data.frame("scenario"=c("Coef=0.1","Coef=0.5","Coef=1.0","Coef=3.0","SD=0.1","SD=0.5","SD=1.0","SD=2.0","Ratio=0.04","Ratio=0.20","Ratio=0.36","Ratio=0.50"),
                      "lambda2"=rep(0,12),
                      "lambda3"=rep(0,12))

ts_mat <- data.frame("scenario"=c("Coef=0.1","Coef=0.5","Coef=1.0","Coef=3.0","SD=0.1","SD=0.5","SD=1.0","SD=2.0","Ratio=0.04","Ratio=0.20","Ratio=0.36","Ratio=0.50"),
                     "SS_learning_tune" = rep(0,12),
                     "SS_learning_opt" = rep(0,12),
                     "OWL_S1" = rep(0,12),
                     "OWL_S" = rep(0,12))
time_mat <- data.frame("scenario"=c("Coef=0.1","Coef=0.5","Coef=1.0","Coef=3.0","SD=0.1","SD=0.5","SD=1.0","SD=2.0","Ratio=0.04","Ratio=0.20","Ratio=0.36","Ratio=0.50"),
                       "SS_learning_tune" = rep(0,12),
                       "OWL_S1" = rep(0,12),
                       "OWL_S" = rep(0,12))
evf_B1_mat <- data.frame("scenario"=c("Coef=0.1","Coef=0.5","Coef=1.0","Coef=3.0","SD=0.1","SD=0.5","SD=1.0","SD=2.0","Ratio=0.04","Ratio=0.20","Ratio=0.36","Ratio=0.50"),
                         "SS_learning_tune" = rep(0,12),
                         "SS_learning_opt" = rep(0,12),
                         "OWL_S1" = rep(0,12),
                         "OWL_S" = rep(0,12))
evf_B2_mat <- data.frame("scenario"=c("Coef=0.1","Coef=0.5","Coef=1.0","Coef=3.0","SD=0.1","SD=0.5","SD=1.0","SD=2.0","Ratio=0.04","Ratio=0.20","Ratio=0.36","Ratio=0.50"),
                         "SS_learning_tune" = rep(0,12),
                         "SS_learning_opt" = rep(0,12),
                         "OWL_S1" = rep(0,12),
                         "OWL_S" = rep(0,12))
evf_B3_mat <- data.frame("scenario"=c("Coef=0.1","Coef=0.5","Coef=1.0","Coef=3.0","SD=0.1","SD=0.5","SD=1.0","SD=2.0","Ratio=0.04","Ratio=0.20","Ratio=0.36","Ratio=0.50"),
                         "SS_learning_tune" = rep(0,12),
                         "SS_learning_opt" = rep(0,12),
                         "OWL_S1" = rep(0,12),
                         "OWL_S" = rep(0,12))

########################################################################################
## comparison varying B3coef ###########################################################
setwd("~/Box/Synergistic_Self_Learning/SS_learning/simulation_20191126_setting2/comparison_B3_varying_coef")
coef_vec <- c(0.1,0.5,1.0,3.0)
num_coef <- length(coef_vec)

acc_indpt_tune <- read.csv("acc_indpt_tune.txt", header = FALSE)
acc_indpt_opt <- read.csv("acc_indpt_opt.txt", header = FALSE)
acc_indpt_1CV <- read.csv("acc_indpt_1CV.txt", header = FALSE)
acc_indpt_allCV <- read.csv("acc_indpt_allCV.txt", header = FALSE)

data <- data.frame("coef"=coef_vec,
                   "acc_indpt_tune"=apply(acc_indpt_tune, 2, mean),
                   "acc_indpt_opt"=apply(acc_indpt_opt, 2, mean),
                   "acc_indpt_1CV"=apply(acc_indpt_1CV, 2, mean),
                   "acc_indpt_allCV"=apply(acc_indpt_allCV, 2, mean))

plot_data <- melt(data, id="coef")
plot_data$variable <- factor(plot_data$variable, 
                             levels = c("acc_indpt_tune","acc_indpt_opt","acc_indpt_1CV","acc_indpt_allCV"))

p11 <- ggplot(data=plot_data, aes(x=coef, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Prediction accuracy of independent data")+
  xlab("Beta3")+
  scale_x_continuous(breaks = seq(0,3,0.5))+
  ylim(0.4,0.91)+
  scale_color_brewer(palette="Spectral",
                     labels=c("SS-learning (tuning)", "SS-learning (oracle)", "OWL on S1", "OWL on S=S1+S2+S3"))+
  theme(legend.position = c(0.8, 0.2), legend.title = element_blank())
p11

################
acc_all_tune <- read.csv("acc_all_tune.txt", header = FALSE)
acc_all_opt <- read.csv("acc_all_opt.txt", header = FALSE)
acc_all_1CV <- read.csv("acc_all_1CV.txt", header = FALSE)
acc_all_allCV <- read.csv("acc_all_allCV.txt", header = FALSE)

data <- data.frame("coef"=coef_vec,
                   "acc_all_tune"=apply(acc_all_tune, 2, mean),
                   "acc_all_opt"=apply(acc_all_opt, 2, mean),
                   "acc_all_1CV"=apply(acc_all_1CV, 2, mean),
                   "acc_all_allCV"=apply(acc_all_allCV, 2, mean))

plot_data <- melt(data, id="coef")
plot_data$variable <- factor(plot_data$variable, 
                             levels = c("acc_all_tune","acc_all_opt","acc_all_1CV","acc_all_allCV"))

p12 <- ggplot(data=plot_data, aes(x=coef, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Prediction accuracy of training data")+
  xlab("Beta3")+
  scale_x_continuous(breaks = seq(0,3,0.5))+
  ylim(0.4,0.91)+
  scale_color_brewer(palette="Spectral",
                     labels=c("SS-learning (tuning)", "SS-learning (oracle)", "OWL on S1", "OWL on S=S1+S2+S3"))+
  theme(legend.position = c(0.8, 0.2), legend.title = element_blank())
p12

################
ts_tune <- read.csv("ts_tune.txt", header = FALSE)
ts_opt <- read.csv("ts_opt.txt", header = FALSE)
ts_1CV <- read.csv("ts_1CV.txt", header = FALSE)
ts_allCV <- read.csv("ts_allCV.txt", header = FALSE)

ts_mat[1:4,2] = sapply(1:num_coef, function(i) paste(format(round(mean(ts_tune[,i]), digits = 2), nsmall=2), " (", format(round(sd(ts_tune[,i]), digits = 2), nsmall=2), ")", sep = ""))
ts_mat[1:4,3] = sapply(1:num_coef, function(i) paste(format(round(mean(ts_opt[,i]), digits = 2), nsmall=2), " (", format(round(sd(ts_opt[,i]), digits = 2), nsmall=2), ")", sep = ""))
ts_mat[1:4,4] = sapply(1:num_coef, function(i) paste(format(round(mean(ts_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(ts_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
ts_mat[1:4,5] = sapply(1:num_coef, function(i) paste(format(round(mean(ts_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(ts_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
time_tune <- read.csv("time_tune.txt", header = FALSE)
time_1CV <- read.csv("time_1CV.txt", header = FALSE)
time_allCV <- read.csv("time_allCV.txt", header = FALSE)

time_mat[1:4,2] = sapply(1:num_coef, function(i) paste(format(round(mean(time_tune[,i]/968), digits = 2),nsmall=2), " (", format(round(sd(time_tune[,i]/968), digits = 2),nsmall=2), ")", sep = ""))
time_mat[1:4,3] = sapply(1:num_coef, function(i) paste(format(round(mean(time_1CV[,i]), digits = 2),nsmall=2), " (", format(round(sd(time_1CV[,i]), digits = 2),nsmall=2), ")", sep = ""))
time_mat[1:4,4] = sapply(1:num_coef, function(i) paste(format(round(mean(time_allCV[,i]), digits = 2),nsmall=2), " (", format(round(sd(time_allCV[,i]), digits = 2),nsmall=2), ")", sep = ""))

################
evf_B1_tune <- read.csv("evf_B1_tune.txt", header = FALSE)
evf_B1_opt <- read.csv("evf_B1_opt.txt", header = FALSE)
evf_B1_1CV <- read.csv("evf_B1_1CV.txt", header = FALSE) 
evf_B1_allCV <- read.csv("evf_B1_allCV.txt", header = FALSE)

evf_B1_mat[1:4,2] = sapply(1:num_coef, function(i) paste(format(round(mean(evf_B1_tune[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B1_tune[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B1_mat[1:4,3] = sapply(1:num_coef, function(i) paste(format(round(mean(evf_B1_opt[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B1_opt[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B1_mat[1:4,4] = sapply(1:num_coef, function(i) paste(format(round(mean(evf_B1_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B1_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B1_mat[1:4,5] = sapply(1:num_coef, function(i) paste(format(round(mean(evf_B1_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B1_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
evf_B2_tune <- read.csv("evf_B2_tune.txt", header = FALSE)
evf_B2_opt <- read.csv("evf_B2_opt.txt", header = FALSE)
evf_B2_1CV <- read.csv("evf_B2_1CV.txt", header = FALSE) 
evf_B2_allCV <- read.csv("evf_B2_allCV.txt", header = FALSE)

evf_B2_mat[1:4,2] = sapply(1:num_coef, function(i) paste(format(round(mean(evf_B2_tune[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B2_tune[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B2_mat[1:4,3] = sapply(1:num_coef, function(i) paste(format(round(mean(evf_B2_opt[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B2_opt[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B2_mat[1:4,4] = sapply(1:num_coef, function(i) paste(format(round(mean(evf_B2_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B2_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B2_mat[1:4,5] = sapply(1:num_coef, function(i) paste(format(round(mean(evf_B2_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B2_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
evf_B3_tune <- read.csv("evf_B3_tune.txt", header = FALSE)
evf_B3_opt <- read.csv("evf_B3_opt.txt", header = FALSE)
evf_B3_1CV <- read.csv("evf_B3_1CV.txt", header = FALSE) 
evf_B3_allCV <- read.csv("evf_B3_allCV.txt", header = FALSE)

evf_B3_mat[1:4,2] = sapply(1:num_coef, function(i) paste(format(round(mean(evf_B3_tune[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B3_tune[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B3_mat[1:4,3] = sapply(1:num_coef, function(i) paste(format(round(mean(evf_B3_opt[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B3_opt[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B3_mat[1:4,4] = sapply(1:num_coef, function(i) paste(format(round(mean(evf_B3_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B3_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B3_mat[1:4,5] = sapply(1:num_coef, function(i) paste(format(round(mean(evf_B3_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B3_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
lam2_tune <- read.csv("lam2_tune.txt", header = FALSE)
lam3_tune <- read.csv("lam3_tune.txt", header = FALSE)

lam_mat[1:4,2] <- sapply(1:num_coef, function(i) paste(format(round(mean(lam2_tune[,i]), digits = 2),nsmall=2)," (",format(round(sd(lam2_tune[,i]), digits = 2),nsmall=2),")", sep = ""))
lam_mat[1:4,3] <- sapply(1:num_coef, function(i) paste(format(round(mean(lam3_tune[,i]), digits = 2),nsmall=2)," (",format(round(sd(lam3_tune[,i]), digits = 2),nsmall=2),")", sep = ""))


########################################################################################
## comparison varying B3sd #############################################################
setwd("~/Box/Synergistic_Self_Learning/SS_learning/simulation_20191126_setting2/comparison_B3_varying_sd")
sd_vec <- c(0.1,0.5,1.0,2.0)
num_sd <- length(sd_vec)

acc_indpt_tune <- read.csv("acc_indpt_tune.txt", header = FALSE)
acc_indpt_opt <- read.csv("acc_indpt_opt.txt", header = FALSE)
acc_indpt_1CV <- read.csv("acc_indpt_1CV.txt", header = FALSE)
acc_indpt_allCV <- read.csv("acc_indpt_allCV.txt", header = FALSE)

data <- data.frame("sd"=sd_vec,
                   "acc_indpt_tune"=apply(acc_indpt_tune, 2, mean),
                   "acc_indpt_opt"=apply(acc_indpt_opt, 2, mean),
                   "acc_indpt_1CV"=apply(acc_indpt_1CV, 2, mean),
                   "acc_indpt_allCV"=apply(acc_indpt_allCV, 2, mean))

plot_data <- melt(data, id="sd")
plot_data$variable <- factor(plot_data$variable, 
                             levels = c("acc_indpt_tune","acc_indpt_opt","acc_indpt_1CV","acc_indpt_allCV"))

p21 <- ggplot(data=plot_data, aes(x=sd, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Prediction accuracy of independent data")+
  xlab("Sigma3")+
  scale_x_continuous(breaks = seq(0,2,0.5))+
  ylim(0.4,0.9)+
  scale_color_brewer(palette="Spectral",
                     labels=c("SS-learning (tuning)", "SS-learning (oracle)", "OWL on S1", "OWL on S=S1+S2+S3"))+
  theme(legend.position = c(0.8, 0.2), legend.title = element_blank())
p21

################
acc_all_tune <- read.csv("acc_all_tune.txt", header = FALSE)
acc_all_opt <- read.csv("acc_all_opt.txt", header = FALSE)
acc_all_1CV <- read.csv("acc_all_1CV.txt", header = FALSE)
acc_all_allCV <- read.csv("acc_all_allCV.txt", header = FALSE)

data <- data.frame("sd"=sd_vec,
                   "acc_all_tune"=apply(acc_all_tune, 2, mean),
                   "acc_all_opt"=apply(acc_all_opt, 2, mean),
                   "acc_all_1CV"=apply(acc_all_1CV, 2, mean),
                   "acc_all_allCV"=apply(acc_all_allCV, 2, mean))

plot_data <- melt(data, id="sd")
plot_data$variable <- factor(plot_data$variable, 
                             levels = c("acc_all_tune","acc_all_opt","acc_all_1CV","acc_all_allCV"))

p22 <- ggplot(data=plot_data, aes(x=sd, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Prediction accuracy of training data")+
  xlab("Sigma3")+
  scale_x_continuous(breaks = seq(0,2,0.5))+
  ylim(0.4,0.9)+
  scale_color_brewer(palette="Spectral",
                     labels=c("SS-learning (tuning)", "SS-learning (oracle)", "OWL on S1", "OWL on S=S1+S2+S3"))+
  theme(legend.position = c(0.8, 0.2), legend.title = element_blank())
p22

################
ts_tune <- read.csv("ts_tune.txt", header = FALSE)
ts_opt <- read.csv("ts_opt.txt", header = FALSE)
ts_1CV <- read.csv("ts_1CV.txt", header = FALSE)
ts_allCV <- read.csv("ts_allCV.txt", header = FALSE)

ts_mat[5:8,2] = sapply(1:num_sd, function(i) paste(format(round(mean(ts_tune[,i]), digits = 2), nsmall=2), " (", format(round(sd(ts_tune[,i]), digits = 2), nsmall=2), ")", sep = ""))
ts_mat[5:8,3] = sapply(1:num_sd, function(i) paste(format(round(mean(ts_opt[,i]), digits = 2), nsmall=2), " (", format(round(sd(ts_opt[,i]), digits = 2), nsmall=2), ")", sep = ""))
ts_mat[5:8,4] = sapply(1:num_sd, function(i) paste(format(round(mean(ts_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(ts_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
ts_mat[5:8,5] = sapply(1:num_sd, function(i) paste(format(round(mean(ts_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(ts_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
time_tune <- read.csv("time_tune.txt", header = FALSE)
time_1CV <- read.csv("time_1CV.txt", header = FALSE)
time_allCV <- read.csv("time_allCV.txt", header = FALSE)

time_mat[5:8,2] = sapply(1:num_coef, function(i) paste(format(round(mean(time_tune[,i]/968), digits = 2), nsmall=2), " (", format(round(sd(time_tune[,i]/968), digits = 2), nsmall=2), ")", sep = ""))
time_mat[5:8,3] = sapply(1:num_coef, function(i) paste(format(round(mean(time_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(time_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
time_mat[5:8,4] = sapply(1:num_coef, function(i) paste(format(round(mean(time_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(time_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
evf_B1_tune <- read.csv("evf_B1_tune.txt", header = FALSE)
evf_B1_opt <- read.csv("evf_B1_opt.txt", header = FALSE)
evf_B1_1CV <- read.csv("evf_B1_1CV.txt", header = FALSE) 
evf_B1_allCV <- read.csv("evf_B1_allCV.txt", header = FALSE)

evf_B1_mat[5:8,2] = sapply(1:num_sd, function(i) paste(format(round(mean(evf_B1_tune[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B1_tune[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B1_mat[5:8,3] = sapply(1:num_sd, function(i) paste(format(round(mean(evf_B1_opt[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B1_opt[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B1_mat[5:8,4] = sapply(1:num_sd, function(i) paste(format(round(mean(evf_B1_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B1_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B1_mat[5:8,5] = sapply(1:num_sd, function(i) paste(format(round(mean(evf_B1_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B1_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
evf_B2_tune <- read.csv("evf_B2_tune.txt", header = FALSE)
evf_B2_opt <- read.csv("evf_B2_opt.txt", header = FALSE)
evf_B2_1CV <- read.csv("evf_B2_1CV.txt", header = FALSE) 
evf_B2_allCV <- read.csv("evf_B2_allCV.txt", header = FALSE)

evf_B2_mat[5:8,2] = sapply(1:num_sd, function(i) paste(format(round(mean(evf_B2_tune[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B2_tune[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B2_mat[5:8,3] = sapply(1:num_sd, function(i) paste(format(round(mean(evf_B2_opt[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B2_opt[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B2_mat[5:8,4] = sapply(1:num_sd, function(i) paste(format(round(mean(evf_B2_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B2_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B2_mat[5:8,5] = sapply(1:num_sd, function(i) paste(format(round(mean(evf_B2_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B2_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
evf_B3_tune <- read.csv("evf_B3_tune.txt", header = FALSE)
evf_B3_opt <- read.csv("evf_B3_opt.txt", header = FALSE)
evf_B3_1CV <- read.csv("evf_B3_1CV.txt", header = FALSE) 
evf_B3_allCV <- read.csv("evf_B3_allCV.txt", header = FALSE)

evf_B3_mat[5:8,2] = sapply(1:num_sd, function(i) paste(format(round(mean(evf_B3_tune[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B3_tune[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B3_mat[5:8,3] = sapply(1:num_sd, function(i) paste(format(round(mean(evf_B3_opt[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B3_opt[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B3_mat[5:8,4] = sapply(1:num_sd, function(i) paste(format(round(mean(evf_B3_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B3_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B3_mat[5:8,5] = sapply(1:num_sd, function(i) paste(format(round(mean(evf_B3_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B3_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
lam2_tune <- read.csv("lam2_tune.txt", header = FALSE)
lam3_tune <- read.csv("lam3_tune.txt", header = FALSE)

lam_mat[5:8,2] <- sapply(1:num_sd, function(i) paste(format(round(mean(lam2_tune[,i]), digits = 2), nsmall=2)," (",format(round(sd(lam2_tune[,i]), digits = 2), nsmall=2),")", sep = ""))
lam_mat[5:8,3] <- sapply(1:num_sd, function(i) paste(format(round(mean(lam3_tune[,i]), digits = 2), nsmall=2)," (",format(round(sd(lam3_tune[,i]), digits = 2), nsmall=2),")", sep = ""))


########################################################################################
## comparison varying n1n3 #############################################################
setwd("~/Box/Synergistic_Self_Learning/SS_learning/simulation_20191126_setting2/comparison_B3_varying_n1n3")
ratio_vec <- c(20,100,180,250)/500
num_ratio <- length(ratio_vec)

acc_indpt_tune <- read.csv("acc_indpt_tune.txt", header = FALSE)
acc_indpt_opt <- read.csv("acc_indpt_opt.txt", header = FALSE)
acc_indpt_1CV <- read.csv("acc_indpt_1CV.txt", header = FALSE)
acc_indpt_allCV <- read.csv("acc_indpt_allCV.txt", header = FALSE)

data <- data.frame("ratio"=ratio_vec,
                   "acc_indpt_tune"=apply(acc_indpt_tune, 2, mean),
                   "acc_indpt_opt"=apply(acc_indpt_opt, 2, mean),
                   "acc_indpt_1CV"=apply(acc_indpt_1CV, 2, mean),
                   "acc_indpt_allCV"=apply(acc_indpt_allCV, 2, mean))

plot_data <- melt(data, id="ratio")
plot_data$variable <- factor(plot_data$variable, 
                             levels = c("acc_indpt_tune","acc_indpt_opt","acc_indpt_1CV","acc_indpt_allCV"))

p31 <- ggplot(data=plot_data, aes(x=ratio, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Prediction accuracy of independent data")+
  xlab("n1/n")+
  scale_x_continuous(breaks = seq(0,0.5,0.1))+
  ylim(0.5,0.9)+
  scale_color_brewer(palette="Spectral",
                     labels=c("SS-learning (tuning)", "SS-learning (oracle)", "OWL on S1", "OWL on S=S1+S2+S3"))+
  theme(legend.position = c(0.8, 0.2), legend.title = element_blank())
p31

################
acc_all_tune <- read.csv("acc_all_tune.txt", header = FALSE)
acc_all_opt <- read.csv("acc_all_opt.txt", header = FALSE)
acc_all_1CV <- read.csv("acc_all_1CV.txt", header = FALSE)
acc_all_allCV <- read.csv("acc_all_allCV.txt", header = FALSE)

data <- data.frame("ratio"=ratio_vec,
                   "acc_all_tune"=apply(acc_all_tune, 2, mean),
                   "acc_all_opt"=apply(acc_all_opt, 2, mean),
                   "acc_all_1CV"=apply(acc_all_1CV, 2, mean),
                   "acc_all_allCV"=apply(acc_all_allCV, 2, mean))

plot_data <- melt(data, id="ratio")
plot_data$variable <- factor(plot_data$variable, 
                             levels = c("acc_all_tune","acc_all_opt","acc_all_1CV","acc_all_allCV"))

p32 <- ggplot(data=plot_data, aes(x=ratio, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Prediction accuracy of independent data")+
  xlab("n1/n")+
  scale_x_continuous(breaks = seq(0,0.5,0.1))+
  ylim(0.5,0.9)+
  scale_color_brewer(palette="Spectral",
                     labels=c("SS-learning (tuning)", "SS-learning (oracle)", "OWL on S1", "OWL on S=S1+S2+S3"))+
  theme(legend.position = c(0.8, 0.2), legend.title = element_blank())
p32

################
ts_tune <- read.csv("ts_tune.txt", header = FALSE)
ts_opt <- read.csv("ts_opt.txt", header = FALSE)
ts_1CV <- read.csv("ts_1CV.txt", header = FALSE)
ts_allCV <- read.csv("ts_allCV.txt", header = FALSE)

ts_mat[9:12,2] = sapply(1:num_ratio, function(i) paste(format(round(mean(ts_tune[,i]), digits = 2), nsmall=2), " (", format(round(sd(ts_tune[,i]), digits = 2), nsmall=2), ")", sep = ""))
ts_mat[9:12,3] = sapply(1:num_ratio, function(i) paste(format(round(mean(ts_opt[,i]), digits = 2), nsmall=2), " (", format(round(sd(ts_opt[,i]), digits = 2), nsmall=2), ")", sep = ""))
ts_mat[9:12,4] = sapply(1:num_ratio, function(i) paste(format(round(mean(ts_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(ts_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
ts_mat[9:12,5] = sapply(1:num_ratio, function(i) paste(format(round(mean(ts_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(ts_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
time_tune <- read.csv("time_tune.txt", header = FALSE)
time_1CV <- read.csv("time_1CV.txt", header = FALSE)
time_allCV <- read.csv("time_allCV.txt", header = FALSE)

time_mat[9:12,2] = sapply(1:num_ratio, function(i) paste(format(round(mean(time_tune[,i]/968), digits = 2), nsmall=2), " (", format(round(sd(time_tune[,i]/968), digits = 2), nsmall=2), ")", sep = ""))
time_mat[9:12,3] = sapply(1:num_ratio, function(i) paste(format(round(mean(time_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(time_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
time_mat[9:12,4] = sapply(1:num_ratio, function(i) paste(format(round(mean(time_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(time_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
evf_B1_tune <- read.csv("evf_B1_tune.txt", header = FALSE)
evf_B1_opt <- read.csv("evf_B1_opt.txt", header = FALSE)
evf_B1_1CV <- read.csv("evf_B1_1CV.txt", header = FALSE) 
evf_B1_allCV <- read.csv("evf_B1_allCV.txt", header = FALSE)

evf_B1_mat[9:12,2] = sapply(1:num_ratio, function(i) paste(format(round(mean(evf_B1_tune[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B1_tune[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B1_mat[9:12,3] = sapply(1:num_ratio, function(i) paste(format(round(mean(evf_B1_opt[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B1_opt[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B1_mat[9:12,4] = sapply(1:num_ratio, function(i) paste(format(round(mean(evf_B1_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B1_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B1_mat[9:12,5] = sapply(1:num_ratio, function(i) paste(format(round(mean(evf_B1_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B1_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
evf_B2_tune <- read.csv("evf_B2_tune.txt", header = FALSE)
evf_B2_opt <- read.csv("evf_B2_opt.txt", header = FALSE)
evf_B2_1CV <- read.csv("evf_B2_1CV.txt", header = FALSE) 
evf_B2_allCV <- read.csv("evf_B2_allCV.txt", header = FALSE)

evf_B2_mat[9:12,2] = sapply(1:num_ratio, function(i) paste(format(round(mean(evf_B2_tune[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B2_tune[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B2_mat[9:12,3] = sapply(1:num_ratio, function(i) paste(format(round(mean(evf_B2_opt[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B2_opt[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B2_mat[9:12,4] = sapply(1:num_ratio, function(i) paste(format(round(mean(evf_B2_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B2_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B2_mat[9:12,5] = sapply(1:num_ratio, function(i) paste(format(round(mean(evf_B2_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B2_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
evf_B3_tune <- read.csv("evf_B3_tune.txt", header = FALSE)
evf_B3_opt <- read.csv("evf_B3_opt.txt", header = FALSE)
evf_B3_1CV <- read.csv("evf_B3_1CV.txt", header = FALSE) 
evf_B3_allCV <- read.csv("evf_B3_allCV.txt", header = FALSE)

evf_B3_mat[9:12,2] = sapply(1:num_ratio, function(i) paste(format(round(mean(evf_B3_tune[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B3_tune[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B3_mat[9:12,3] = sapply(1:num_ratio, function(i) paste(format(round(mean(evf_B3_opt[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B3_opt[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B3_mat[9:12,4] = sapply(1:num_ratio, function(i) paste(format(round(mean(evf_B3_1CV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B3_1CV[,i]), digits = 2), nsmall=2), ")", sep = ""))
evf_B3_mat[9:12,5] = sapply(1:num_ratio, function(i) paste(format(round(mean(evf_B3_allCV[,i]), digits = 2), nsmall=2), " (", format(round(sd(evf_B3_allCV[,i]), digits = 2), nsmall=2), ")", sep = ""))

################
lam2_tune <- read.csv("lam2_tune.txt", header = FALSE)
lam3_tune <- read.csv("lam3_tune.txt", header = FALSE)

lam_mat[9:12,2] <- sapply(1:num_ratio, function(i) paste(format(round(mean(lam2_tune[,i]), digits = 2), nsmall=2)," (",format(round(sd(lam2_tune[,i]), digits = 2), nsmall=2),")", sep = ""))
lam_mat[9:12,3] <- sapply(1:num_ratio, function(i) paste(format(round(mean(lam3_tune[,i]), digits = 2), nsmall=2)," (",format(round(sd(lam3_tune[,i]), digits = 2), nsmall=2),")", sep = ""))

########################################################################################
########################################################################################
setwd("~/Box/Synergistic_Self_Learning/SS_learning/JASA_paper_submission/results")

png("Figure4_comparison.png", height = 12, width = 10, res = 300, units = "in")
plot_grid(p12,p11,p22,p21,p32,p31, labels = c("a","","b","","c",""), ncol = 2)
dev.off()

write.table(lam_mat, file = "Table1_lam_mat.txt", col.names = TRUE, row.names = FALSE, sep = ",")
write.table(ts_mat, file = "Table5_ts_mat.txt", col.names = TRUE, row.names = FALSE, sep = ",")
write.table(time_mat, file = "Table6_time_mat.txt", col.names = TRUE, row.names = FALSE, sep = ",")
write.table(evf_B1_mat, file = "Table7_evf_B1_mat.txt", col.names = TRUE, row.names = FALSE, sep = ",")
write.table(evf_B2_mat, file = "Table8_evf_B2_mat.txt", col.names = TRUE, row.names = FALSE, sep = ",")
write.table(evf_B3_mat, file = "Table9_evf_B3_mat.txt", col.names = TRUE, row.names = FALSE, sep = ",")
