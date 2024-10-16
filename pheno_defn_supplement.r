## definition of edu years

## age completed full time education
fte_tmp <- as.matrix(edu)[,rep('f.845.0.0',6)]
no_school <- (fte_tmp==-2) 
fte_tmp[fte_tmp==-2] <- NA ## Never went to school
fte_tmp[fte_tmp==-1] <- NA ## Do not know
fte_tmp[fte_tmp==-3] <- NA ## Prefer not to answer

## eduyears
edu_tmp <- as.matrix(edu)[,paste0('f.6138.0.',0:5)]
edu_tmp[edu_tmp %in% c(-3)] <- NA
edu_tmp[edu_tmp==1] <- 20  ## College or University degree
edu_tmp[edu_tmp==2] <- 13  ## A levels/AS levels or equivalent
edu_tmp[edu_tmp==3] <- 10  ## O levels/GCSEs or equivalent
edu_tmp[edu_tmp==4] <- 10  ## CSEs or equivalent
edu_tmp[edu_tmp==6] <- 15  ## Other professional qualifications eg: nursing, teaching
edu_tmp[edu_tmp==-7] <- 7   ## None of the above
edu_tmp[!is.na(edu_tmp) & edu_tmp==5 & !is.na(fte_tmp)] <- 
    fte_tmp[!is.na(edu_tmp) & edu_tmp==5& !is.na(fte_tmp)] - 5
    ## NVQ or HND or HNC or equivalent
edu_tmp[no_school] <- 0
edu_tmp[edu_tmp>20] <- 20
edu_tmp[edu_tmp<0] <- 0

eduyears_visit_0 <- apply(edu_tmp, 1, max, na.rm=T)
eduyears_visit_0[eduyears_visit_0==-Inf] <- NA