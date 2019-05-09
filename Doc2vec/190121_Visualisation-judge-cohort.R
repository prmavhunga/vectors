library(ggplot2)
library(stringi)
library(plyr)
library(stringr)

#data for B1
CB_demeaned_vectors_for_judges <- read.csv('CB_demeaned_vectors_for_judges.csv')

CB_demeaned_vectors_for_judges_cohort <- CB_demeaned_vectors_for_judges[CB_demeaned_vectors_for_judges$cohort != '',]
CB_demeaned_vectors_for_judges_cohort_cc <- CB_demeaned_vectors_for_judges_cohort[CB_demeaned_vectors_for_judges_cohort$circuit != 'Supreme Court',]
ch_break <- levels(CB_demeaned_vectors_for_judges_cohort_cc$cohort)
ch_label <- c('','10s','20s','30s','40s','50s')
CB_demeaned_vectors_for_judges_cohort_cc$ch <- CB_demeaned_vectors_for_judges_cohort_cc$cohort
CB_demeaned_vectors_for_judges_cohort_cc$ch <- factor(CB_demeaned_vectors_for_judges_cohort_cc$ch, levels=ch_break, labels=ch_label)

plot6 <- ggplot(data = CB_demeaned_vectors_for_judges_cohort_cc, aes(x=x, y=y, label = ch, color=cohort)) + 
  geom_text(size = 3, alpha = 0.8) +
  ggtitle("Cohort, CC Judge Vector, Demeaned by Circuit and Big Topic")
ggsave("plotB1.pdf", plot = plot6, width = 40, height = 40, units = "cm", limitsize = FALSE)


