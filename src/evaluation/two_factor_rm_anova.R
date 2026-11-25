#install.packages("rstatix")
#install.packages("ggpubr")
#install.packages("readr")
#install.packages("dplyr")
#install.packages("lme4")
library(rstatix)
library(readr)
library(dplyr)
library(ggpubr)
library(lme4)


# ############################################ Load, preprocess ############################################
args <- commandArgs(trailingOnly=TRUE)
input_path <- args[1]
output_path <- args[2]
df <- read_csv(input_path)

dv <- "abs_gain"
df <- select(df, run, homophily_requested, label_rate, dv)

df$subject <- as.factor(df$run)
df$homophily <- as.factor(df$homophily_requested)
df$label_rate <- as.factor(df$label_rate)

cat("\n\n=== Filtered df ===\n")
df <- select(df, subject, homophily, label_rate, dv)
str(df)
write.csv(df, file=file.path(output_path, "df.csv"), row.names=FALSE)
cat("\n\n")

cat("=== Cell (h,lr) sizes ===\n")
df_grouped <- group_by(df, homophily, label_rate)
group_size(df_grouped) 
cat("\n\n")

# ############################################ Assumptions ############################################
# Assumption 1 - no outliers within (h,lr) cells
cat("=== Outliers ===\n")
outliers <- identify_outliers(data=df_grouped, variable=dv)
outliers # we care more about extreme outliers
write.csv(outliers, file=file.path(output_path, "outliers.csv"), row.names=FALSE)
cat("\n\n")

# Assumption 2 - dependent variable normally distributed for each observation
cat("=== Normality test ===\n")
test <- shapiro_test(data=df_grouped, variables=dv) 
test
cat("=== Filtered (p<0.05) ===\n")
normality_violations <- filter(test, p<0.05)
normality_violations

write.csv(test, file=file.path(output_path, "normality.csv"), row.names=FALSE)
cat("\n\n")

df_qq <- filter(
  df,
  homophily %in% c("0", "0.2", "0.4", "0.6", "0.8", "1"),
  label_rate %in% c("0.20", "0.15", "0.10", "0.05", "0.01")
)

qq_plot <- ggqqplot(df_qq, dv, ggtheme=theme_light()) +
  facet_grid(label_rate~homophily, labeller="label_both") # rows~cols
ggsave(filename=file.path(output_path, "qq_plot.png"), width=10, height=10, plot=qq_plot, dpi=300)

# Assumption 3 - sphericity (variance of difference btw groups in factors is equal, 
cat("=== Two-way repeated measures ANOVA ===\n")
res <- anova_test(
  data=df, 
  dv=dv,
  wid=subject, # subject id
  within=c(homophily, label_rate)
)

res_table <- get_anova_table(res)
res_table

write.csv(res_table, file=file.path(output_path, "two_way_rm_anova.csv"),  row.names=FALSE)
cat("\n\n")

# ############################################ Fetch coefficients ############################################
# Explicitely fit the defined linear model - just to confirm estiamte coefficients actually model trend
cat("=== Model coefficients ===\n")
model <- lmer(abs_gain ~ homophily * label_rate + (1|subject), data=df)
summary(model)
fixed_effects <- as.data.frame(fixef(model))
random_effects <- as.data.frame(ranef(model))
write.csv(fixed_effects, file=file.path(output_path, "fixed_effects_coef.csv"),  row.names=TRUE)
write.csv(random_effects, file=file.path(output_path, "random_effects_coef.csv"),  row.names=TRUE)
cat("\n\n")


# ############################################ Post-hoc ############################################
# If interaction is significant, for which homophily levels does label rate matters? 
#   compare the lr means within each h group
cat("=== One-way repeated measures ANOVA for lr (on each h) ===\n")
df_grouped_h <- group_by(df, homophily)

res_h <- anova_test(
  data=df_grouped_h, 
  dv=dv,
  wid=subject, # subject id
  within=c(label_rate)
)

res_h_table <- get_anova_table(res_h)
res_h_table
write.csv(res_h_table, file=file.path(output_path, "one_way_rm_anova_lr.csv"), row.names=FALSE)
cat("\n\n")



# ############################################ Sensitivity Analysis ############################################
extreme_outliers <- filter(outliers, is.extreme==TRUE)
df_no_outliers <- anti_join(df,extreme_outliers, by=c("subject", "homophily", "label_rate"))

write.csv(df_no_outliers, file=file.path(output_path, "df_no_outliers.csv"), row.names=FALSE)

cat("=== Sensitivity analysis ===\n")
res_no_outliers <- anova_test(
  data=df_no_outliers, 
  dv=dv,
  wid=subject, # subject id
  within=c(homophily, label_rate)
)

res_table_no_outliers <- get_anova_table(res_no_outliers)
res_table_no_outliers

write.csv(res_table_no_outliers, file=file.path(output_path, "two_way_rm_anova_sensitivity.csv"),  row.names=FALSE)
cat("\n\n")