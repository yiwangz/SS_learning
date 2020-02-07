library(ggplot2)
library(reshape2)
library(cowplot)

########################################################################################
## convergence varying B3coef###########################################################
setwd("~/Box/Synergistic_Self_Learning/SS_learning/simulation_20191126_setting2/convergence_B3_varying_coef")

coef_vec <- c(0.1,0.5,1.0,3.0)
num_coef <- length(coef_vec)
K <- 20

acc_all_tune_coef <- read.csv("acc_all_tune.txt", header = FALSE)
obj_tune_coef <- read.csv("obj_tune.txt", header = FALSE)
ts_tune_coef <- read.csv("ts_tune.txt", header = FALSE)

diff_acc_all_tune <- sapply(c(2:K,(K+2):(2*K),(2*K+2):(3*K),(3*K+2):(4*K)), function(i) (acc_all_tune_coef[,i]-acc_all_tune_coef[,i-1]))
diff_obj_tune <- sapply(c(2:K,(K+2):(2*K),(2*K+2):(3*K),(3*K+2):(4*K)), function(i) (obj_tune_coef[,i]-obj_tune_coef[,i-1]))
diff_ts_tune <- sapply(c(2:K,(K+2):(2*K),(2*K+2):(3*K),(3*K+2):(4*K)), function(i) (ts_tune_coef[,i]-ts_tune_coef[,i-1]))


####
data_mean_acc_all_tune <- data.frame("iteration"=seq(2,K,1),
                                     "acc_coef1"=apply(diff_acc_all_tune[,1:(K-1)], 2, mean),
                                     "acc_coef2"=apply(diff_acc_all_tune[,K:(2*K-2)], 2, mean),
                                     "acc_coef3"=apply(diff_acc_all_tune[,(2*K-1):(3*K-3)], 2, mean),
                                     "acc_coef4"=apply(diff_acc_all_tune[,(3*K-2):(4*K-4)], 2, mean))
plot_data_mean_acc_all_tune <- melt(data_mean_acc_all_tune, id="iteration")

plot_data_mean_acc_all_tune$variable <- factor(plot_data_mean_acc_all_tune$variable,
                                               levels = c("acc_coef1","acc_coef2","acc_coef3","acc_coef4"))

p11 <- ggplot(data=plot_data_mean_acc_all_tune, aes(x=iteration, y=value, color=variable))+
  geom_line()+
  theme_classic()+
  ylab("Difference of prediction accuracy on training data")+
  xlab("Iteration")+
  scale_x_continuous(breaks = seq(2,20,1))+
  scale_y_continuous(labels = function(x) format(x, scientific = TRUE, digits = 1))+
  scale_color_brewer(palette="Spectral",
                     labels=c("Beta3=0.1", "Beta3=0.5", "Beta3=1.0", "Beta3=3.0"))+
  theme(legend.position = c(0.8, 0.7), legend.title = element_blank())
p11

####
data_mean_obj_tune <- data.frame("iteration"=seq(2,K,1),
                                 "obj_coef1"=apply(diff_obj_tune[,1:(K-1)], 2, mean),
                                 "obj_coef2"=apply(diff_obj_tune[,K:(2*K-2)], 2, mean),
                                 "obj_coef3"=apply(diff_obj_tune[,(2*K-1):(3*K-3)], 2, mean),
                                 "obj_coef4"=apply(diff_obj_tune[,(3*K-2):(4*K-4)], 2, mean))
plot_data_mean_obj_tune <- melt(data_mean_obj_tune, id="iteration")

plot_data_mean_obj_tune$variable <- factor(plot_data_mean_obj_tune$variable, 
                                           levels = c("obj_coef1","obj_coef2","obj_coef3","obj_coef4"))

p12 <- ggplot(data=plot_data_mean_obj_tune, aes(x=iteration, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Difference of objective function")+
  xlab("Iteration")+
  scale_x_continuous(breaks = seq(2,20,1))+
  scale_y_continuous(labels = function(x) format(x, scientific = TRUE, digits = 1))+
  scale_color_brewer(palette="Spectral",
                     labels=c("Beta3=0.1", "Beta3=0.5", "Beta3=1.0", "Beta3=3.0"))+
  theme(legend.position = c(0.8, 0.7), legend.title = element_blank())
p12

####
data_mean_ts_tune <- data.frame("iteration"=seq(2,K,1),
                                "ts_coef1"=apply(diff_ts_tune[,1:(K-1)], 2, mean),
                                "ts_coef2"=apply(diff_ts_tune[,K:(2*K-2)], 2, mean),
                                "ts_coef3"=apply(diff_ts_tune[,(2*K-1):(3*K-3)], 2, mean),
                                "ts_coef4"=apply(diff_ts_tune[,(3*K-2):(4*K-4)], 2, mean))
plot_data_mean_ts_tune <- melt(data_mean_ts_tune, id="iteration")

plot_data_mean_ts_tune$variable <- factor(plot_data_mean_ts_tune$variable, 
                                          levels = c("ts_coef1","ts_coef2","ts_coef3","ts_coef4"))

p13 <- ggplot(data=plot_data_mean_ts_tune, aes(x=iteration, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Difference of sum of squared errors")+
  xlab("Iteration")+
  scale_x_continuous(breaks = seq(2,20,1))+
  scale_y_continuous(labels = function(x) format(x, scientific = TRUE, digits = 1))+
  scale_color_brewer(palette="Spectral",
                     labels=c("Beta3=0.1", "Beta3=0.5", "Beta3=1.0", "Beta3=3.0"))+
  theme(legend.position = c(0.8, 0.7), legend.title = element_blank())
p13

########################################################################################
## convergence varying B3sd#############################################################
setwd("~/Box/Synergistic_Self_Learning/SS_learning/simulation_20191126_setting2/convergence_B3_varying_sd")

sd_vec <- c(0.1,0.5,1.0,2.0)
num_sd <- length(sd_vec)
K <- 20

acc_all_tune_sd <- read.csv("acc_all_tune.txt", header = FALSE)
obj_tune_sd <- read.csv("obj_tune.txt", header = FALSE)
ts_tune_sd <- read.csv("ts_tune.txt", header = FALSE)

diff_acc_all_tune <- sapply(c(2:K,(K+2):(2*K),(2*K+2):(3*K),(3*K+2):(4*K)), function(i) (acc_all_tune_sd[,i]-acc_all_tune_sd[,i-1]))
diff_obj_tune <- sapply(c(2:K,(K+2):(2*K),(2*K+2):(3*K),(3*K+2):(4*K)), function(i) (obj_tune_sd[,i]-obj_tune_sd[,i-1]))
diff_ts_tune <- sapply(c(2:K,(K+2):(2*K),(2*K+2):(3*K),(3*K+2):(4*K)), function(i) (ts_tune_sd[,i]-ts_tune_sd[,i-1]))

####
data_mean_acc_all_tune <- data.frame("iteration"=seq(2,K,1),
                                     "acc_sd1"=apply(diff_acc_all_tune[,1:(K-1)], 2, mean),
                                     "acc_sd2"=apply(diff_acc_all_tune[,K:(2*K-2)], 2, mean),
                                     "acc_sd3"=apply(diff_acc_all_tune[,(2*K-1):(3*K-3)], 2, mean),
                                     "acc_sd4"=apply(diff_acc_all_tune[,(3*K-2):(4*K-4)], 2, mean))
plot_data_mean_acc_all_tune <- melt(data_mean_acc_all_tune, id="iteration")

plot_data_mean_acc_all_tune$variable <- factor(plot_data_mean_acc_all_tune$variable,
                                               levels = c("acc_sd1","acc_sd2","acc_sd3","acc_sd4"))

p21 <- ggplot(data=plot_data_mean_acc_all_tune, aes(x=iteration, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Difference of prediction accuracy on training data")+
  xlab("Iteration")+
  scale_x_continuous(breaks = seq(2,20,1))+
  scale_y_continuous(labels = function(x) format(x, scientific = TRUE, digits = 1))+
  scale_color_brewer(palette="Spectral",
                     labels=c("Beta3=0.1", "Beta3=0.5", "Beta3=1.0", "Beta3=3.0"))+
  theme(legend.position = c(0.8, 0.7), legend.title = element_blank())
p21

####
data_mean_obj_tune <- data.frame("iteration"=seq(2,K,1),
                                 "obj_sd1"=apply(diff_obj_tune[,1:(K-1)], 2, mean),
                                 "obj_sd2"=apply(diff_obj_tune[,K:(2*K-2)], 2, mean),
                                 "obj_sd3"=apply(diff_obj_tune[,(2*K-1):(3*K-3)], 2, mean),
                                 "obj_sd4"=apply(diff_obj_tune[,(3*K-2):(4*K-4)], 2, mean))
plot_data_mean_obj_tune <- melt(data_mean_obj_tune, id="iteration")

plot_data_mean_obj_tune$variable <- factor(plot_data_mean_obj_tune$variable, 
                                           levels = c("obj_sd1","obj_sd2","obj_sd3","obj_sd4"))

p22 <- ggplot(data=plot_data_mean_obj_tune, aes(x=iteration, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Difference of objective function")+
  xlab("Iteration")+
  scale_x_continuous(breaks = seq(2,20,1))+
  scale_y_continuous(labels = function(x) format(x, scientific = TRUE, digits = 1))+
  scale_color_brewer(palette="Spectral",
                     labels=c("Beta3=0.1", "Beta3=0.5", "Beta3=1.0", "Beta3=3.0"))+
  theme(legend.position = c(0.8, 0.7), legend.title = element_blank())
p22

####
data_mean_ts_tune <- data.frame("iteration"=seq(2,K,1),
                                "ts_sd1"=apply(diff_ts_tune[,1:(K-1)], 2, mean),
                                "ts_sd2"=apply(diff_ts_tune[,K:(2*K-2)], 2, mean),
                                "ts_sd3"=apply(diff_ts_tune[,(2*K-1):(3*K-3)], 2, mean),
                                "ts_sd4"=apply(diff_ts_tune[,(3*K-2):(4*K-4)], 2, mean))
plot_data_mean_ts_tune <- melt(data_mean_ts_tune, id="iteration")

plot_data_mean_ts_tune$variable <- factor(plot_data_mean_ts_tune$variable, 
                                          levels = c("ts_sd1","ts_sd2","ts_sd3","ts_sd4"))

p23 <- ggplot(data=plot_data_mean_ts_tune, aes(x=iteration, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Difference of sum of squared errors")+
  xlab("Iteration")+
  scale_x_continuous(breaks = seq(2,20,1))+
  scale_y_continuous(labels = function(x) format(x, scientific = TRUE, digits = 1))+
  scale_color_brewer(palette="Spectral",
                     labels=c("Beta3=0.1", "Beta3=0.5", "Beta3=1.0", "Beta3=3.0"))+
  theme(legend.position = c(0.8, 0.7), legend.title = element_blank())
p23

########################################################################################
## convergence varying n1n3#############################################################
setwd("~/Box/Synergistic_Self_Learning/SS_learning/simulation_20191126_setting2/convergence_B3_varying_n1n3")

ratio_vec <- c(20,100,180,250)/500
num_ratio <- length(ratio_vec)
K <- 20

acc_all_tune_ratio <- read.csv("acc_all_tune.txt", header = FALSE)
obj_tune_ratio <- read.csv("obj_tune.txt", header = FALSE)
ts_tune_ratio <- read.csv("ts_tune.txt", header = FALSE)

diff_acc_all_tune <- sapply(c(2:K,(K+2):(2*K),(2*K+2):(3*K),(3*K+2):(4*K)), function(i) (acc_all_tune_ratio[,i]-acc_all_tune_ratio[,i-1]))
diff_obj_tune <- sapply(c(2:K,(K+2):(2*K),(2*K+2):(3*K),(3*K+2):(4*K)), function(i) (obj_tune_ratio[,i]-obj_tune_ratio[,i-1]))
diff_ts_tune <- sapply(c(2:K,(K+2):(2*K),(2*K+2):(3*K),(3*K+2):(4*K)), function(i) (ts_tune_ratio[,i]-ts_tune_ratio[,i-1]))

####
data_mean_acc_all_tune <- data.frame("iteration"=seq(2,K,1),
                                     "acc_ratio1"=apply(diff_acc_all_tune[,1:(K-1)], 2, mean),
                                     "acc_ratio2"=apply(diff_acc_all_tune[,K:(2*K-2)], 2, mean),
                                     "acc_ratio3"=apply(diff_acc_all_tune[,(2*K-1):(3*K-3)], 2, mean),
                                     "acc_ratio4"=apply(diff_acc_all_tune[,(3*K-2):(4*K-4)], 2, mean))
plot_data_mean_acc_all_tune <- melt(data_mean_acc_all_tune, id="iteration")

plot_data_mean_acc_all_tune$variable <- factor(plot_data_mean_acc_all_tune$variable,
                                               levels = c("acc_ratio1","acc_ratio2","acc_ratio3","acc_ratio4"))

p31 <- ggplot(data=plot_data_mean_acc_all_tune, aes(x=iteration, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Difference of prediction accuracy on training data")+
  xlab("Iteration")+
  scale_x_continuous(breaks = seq(2,20,1))+
  scale_y_continuous(labels = function(x) format(x, scientific = TRUE, digits = 1))+
  scale_color_brewer(palette="Spectral",
                     labels=c("Beta3=0.1", "Beta3=0.5", "Beta3=1.0", "Beta3=3.0"))+
  theme(legend.position = c(0.8, 0.7), legend.title = element_blank())
p31

####
data_mean_obj_tune <- data.frame("iteration"=seq(2,K,1),
                                 "obj_ratio1"=apply(diff_obj_tune[,1:(K-1)], 2, mean),
                                 "obj_ratio2"=apply(diff_obj_tune[,K:(2*K-2)], 2, mean),
                                 "obj_ratio3"=apply(diff_obj_tune[,(2*K-1):(3*K-3)], 2, mean),
                                 "obj_ratio4"=apply(diff_obj_tune[,(3*K-2):(4*K-4)], 2, mean))
plot_data_mean_obj_tune <- melt(data_mean_obj_tune, id="iteration")

plot_data_mean_obj_tune$variable <- factor(plot_data_mean_obj_tune$variable, 
                                           levels = c("obj_ratio1","obj_ratio2","obj_ratio3","obj_ratio4"))

p32 <- ggplot(data=plot_data_mean_obj_tune, aes(x=iteration, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Difference of objective function")+
  xlab("Iteration")+
  scale_x_continuous(breaks = seq(2,20,1))+
  scale_y_continuous(labels = function(x) format(x, scientific = TRUE, digits = 1))+
  scale_color_brewer(palette="Spectral",
                     labels=c("Beta3=0.1", "Beta3=0.5", "Beta3=1.0", "Beta3=3.0"))+
  theme(legend.position = c(0.8, 0.7), legend.title = element_blank())
p32

####
data_mean_ts_tune <- data.frame("iteration"=seq(2,K,1),
                                "ts_ratio1"=apply(diff_ts_tune[,1:(K-1)], 2, mean),
                                "ts_ratio2"=apply(diff_ts_tune[,K:(2*K-2)], 2, mean),
                                "ts_ratio3"=apply(diff_ts_tune[,(2*K-1):(3*K-3)], 2, mean),
                                "ts_ratio4"=apply(diff_ts_tune[,(3*K-2):(4*K-4)], 2, mean))
plot_data_mean_ts_tune <- melt(data_mean_ts_tune, id="iteration")

plot_data_mean_ts_tune$variable <- factor(plot_data_mean_ts_tune$variable, 
                                          levels = c("ts_ratio1","ts_ratio2","ts_ratio3","ts_ratio4"))

p33 <- ggplot(data=plot_data_mean_ts_tune, aes(x=iteration, y=value, color=variable)) +
  geom_line()+
  theme_classic()+
  ylab("Difference of sum of squared errors")+
  xlab("Iteration")+
  scale_x_continuous(breaks = seq(2,20,1))+
  scale_y_continuous(labels = function(x) format(x, scientific = TRUE, digits = 1))+
  scale_color_brewer(palette="Spectral",
                     labels=c("Beta3=0.1", "Beta3=0.5", "Beta3=1.0", "Beta3=3.0"))+
  theme(legend.position = c(0.8, 0.7), legend.title = element_blank())
p33

setwd("~/Box/Synergistic_Self_Learning/SS_learning/JASA_paper_submission/results")
png("Figure3_convergence.png", height = 13.5, width = 11, res = 300, units = "in")
plot_grid(p11, p12, p13, p21, p22, p23, p31, p32, p33, ncol = 3, labels = c("a","","","b","","","c","",""))
dev.off()
