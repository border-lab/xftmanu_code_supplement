import xftsim as xft
import numpy as np
import pandas as pd
import scipy.stats as stats
TIMELIMIT=360
MATEBATCH=1000
THREADS=12 
SEED=1
SUBSIZE=8000

n=16000
m=2000

ddlist1 = np.concatenate([np.arange(k) for k in range(1,6)])
ddlist2 = np.concatenate([np.repeat(k,k) for k in range(1,6)])




for SEED in range(1,51):
    print(f'SEED = {SEED}')

    np.random.seed(SEED)
    psychdat=pd.read_csv('~/psych_cors.csv')
    dx = ["ADHD", "ALC", "ANX", "BIP", "MDD", "SCZ"]
    prev = np.array([.087, .291, .316, .025, .144, .04])
    thresh = stats.norm.ppf(1-prev)

    sdat = psychdat.loc[psychdat.seed==SEED]
    R_mate = np.array([[sdat.rmate.loc[(sdat.dx1 == x) * (sdat.dx2 == y)].values[0] for x in dx] for y in dx])
    h2 = sdat.vg1.groupby(sdat.dx1).mean().values

    mate_batch_size =MATEBATCH


    ####################### SETUP SIM #######################


    founders = xft.founders.founder_haplotypes_uniform_AFs(n=n,m=m,minMAF=.05)
    rmap =xft.reproduce.RecombinationMap.constant_map_from_haplotypes(founders, p = 50/m)

    vg=h2
    ve=1-vg
    phenos = dx


    genetic_ind = xft.index.ComponentIndex.from_product(phenos,'addGen')
    noise_ind = xft.index.ComponentIndex.from_product(phenos,'noise')
    pheno_ind = xft.index.ComponentIndex.from_product(phenos,'phenotype')
    dx_ind = xft.index.ComponentIndex.from_product(phenos,'diagnosis')

    eff = xft.effect.NonOverlappingEffects(vg = h2,
                                           variant_indexer=founders.xft.get_variant_indexer(),
                                           component_indexer=genetic_ind)
    genetic_comp = xft.arch.AdditiveGeneticComponent(eff)
    noise_comp = xft.arch.AdditiveNoiseComponent(variances=ve,
                                                 component_index=noise_ind)

    strans = xft.arch.SumAllTransformation(input_cindex=genetic_ind.merge(noise_ind),
                                            output_component_name='phenotype')
    btrans = xft.arch.BinarizingTransformation(thresholds=thresh,
                                              input_cindex=pheno_ind,
                                              output_cindex=dx_ind,
                                              )

    arch = xft.arch.Architecture([genetic_comp,noise_comp,strans,btrans])


    reg_asrt = xft.mate.GeneralAssortativeMatingRegime(offspring_per_pair=2, 
                                                       mates_per_female=2,
                                                       cross_corr=R_mate,
                                                       component_index=pheno_ind,
                                                       control={'time_limit':TIMELIMIT,
                                                                'nb_threads':THREADS})

    reg_batched = xft.mate.BatchedMatingRegime(regime=reg_asrt,
                                               max_batch_size = mate_batch_size)
    
    sample_stats =[xft.stats.SampleStatistics(), 
        xft.stats.HasemanElstonEstimator(filter_sample=True,
                                         component_index = pheno_ind.merge(dx_ind)),
        xft.stats.MatingStatistics(),]
    estimators = [
        xft.stats.Sib_GWAS_Estimator(n_sub=SUBSIZE),
        xft.stats.Pop_GWAS_Estimator(n_sub=SUBSIZE),
    ]

    sim = xft.sim.Simulation(architecture=arch,
                               founder_haplotypes=founders,
                               mating_regime=reg_batched,
                               statistics=sample_stats+estimators,
                               sample_filter=xft.index.RandomSiblingSubsampleFilter(k=SUBSIZE),
                               recombination_map=rmap,
                               filter_sample=True,
                               )

    for GEN in range(6):
        sim.increment_generation()
        sim.reproduce()
        sim.compute_phenotypes()
        sim.statistics = estimators
        sim.estimate_statistics()
        sim.haplotypes = sim.haplotypes[::2,:]
        sim.phenotypes = sim.phenotypes[::2,:]
        sim.mate()
        sim.update_pedigree()
        sim.apply_filter()
        sim.statistics = sample_stats
        sim.estimate_statistics()
        sim.process()

    sim.pickle_results(f'/data/sres/psy6way_{SEED}',
                       mating_regime=False,
                       metadata={'R_mate':R_mate,'h2':h2})

    for (dd1, dd2) in zip(ddlist1,ddlist2):
        print(f'SEED = {SEED}; dd1 = {dd1}; dd2 = {dd2}')
        ddinds = np.array([dd1,dd2])
        np.random.seed(SEED)
        psychdat=pd.read_csv('~/psych_cors.csv')
        dx = ["ADHD", "ALC", "ANX", "BIP", "MDD", "SCZ"]
        dx_sub = [dx[i] for i in ddinds]
        prev = np.array([.087, .291, .316, .025, .144, .04])
        thresh = stats.norm.ppf(1-prev)

        sdat = psychdat.loc[psychdat.seed==SEED]
        R_mate = np.array([[sdat.rmate.loc[(sdat.dx1 == x) * (sdat.dx2 == y)].values[0] for x in dx_sub] for y in dx_sub])
        h2 = sdat.vg1.groupby(sdat.dx1).mean().values

        ####################### SETUP SIM #######################


        founders = xft.founders.founder_haplotypes_uniform_AFs(n=n,m=m,minMAF=.05)
        rmap =xft.reproduce.RecombinationMap.constant_map_from_haplotypes(founders, p = 50/m)

        vg=h2
        ve=1-vg
        phenos = dx
        print(dd1)
        print(dd2)

        genetic_ind = xft.index.ComponentIndex.from_product(phenos,'addGen')
        noise_ind = xft.index.ComponentIndex.from_product(phenos,'noise')
        pheno_ind = xft.index.ComponentIndex.from_product(phenos,'phenotype')
        mate_ind = xft.index.ComponentIndex.from_product([phenos[dd1], phenos[dd2]],'phenotype')
        dx_ind = xft.index.ComponentIndex.from_product(phenos,'diagnosis')

        eff = xft.effect.NonOverlappingEffects(vg = h2,
                                               variant_indexer=founders.xft.get_variant_indexer(),
                                               component_indexer=genetic_ind)
        genetic_comp = xft.arch.AdditiveGeneticComponent(eff)
        noise_comp = xft.arch.AdditiveNoiseComponent(variances=ve,
                                                     component_index=noise_ind)

        strans = xft.arch.SumAllTransformation(input_cindex=genetic_ind.merge(noise_ind),
                                                output_component_name='phenotype')
        btrans = xft.arch.BinarizingTransformation(thresholds=thresh,
                                                  input_cindex=pheno_ind,
                                                  output_cindex=dx_ind,
                                                  )

        arch = xft.arch.Architecture([genetic_comp,noise_comp,strans,btrans])


        reg_asrt = xft.mate.GeneralAssortativeMatingRegime(offspring_per_pair=2, 
                                                           mates_per_female=2,
                                                           cross_corr=R_mate,
                                                           component_index=mate_ind,
                                                           control={'time_limit':60})

        reg_batched = xft.mate.BatchedMatingRegime(regime=reg_asrt,
                                                   max_batch_size = mate_batch_size)

        sample_stats =[xft.stats.SampleStatistics(), 
            xft.stats.HasemanElstonEstimator(filter_sample=True,
                                             component_index = pheno_ind.merge(dx_ind)),
            xft.stats.MatingStatistics(),]
        estimators = [
            xft.stats.Sib_GWAS_Estimator(n_sub=SUBSIZE),
            xft.stats.Pop_GWAS_Estimator(n_sub=SUBSIZE),
        ]

        sim = xft.sim.Simulation(architecture=arch,
                                   founder_haplotypes=founders,
                                   mating_regime=reg_batched,
                                   statistics=sample_stats+estimators,
                                   sample_filter=xft.index.RandomSiblingSubsampleFilter(k=SUBSIZE),
                                   recombination_map=rmap,
                                   filter_sample=True,
                                   )

        for GEN in range(6):
            sim.increment_generation()
            sim.reproduce()
            sim.compute_phenotypes()
            sim.statistics = estimators
            sim.estimate_statistics()
            sim.haplotypes = sim.haplotypes[::2,:]
            sim.phenotypes = sim.phenotypes[::2,:]
            sim.mate()
            sim.update_pedigree()
            sim.apply_filter()
            sim.statistics = sample_stats
            sim.estimate_statistics()
            sim.process()

        sim.pickle_results(f'/data/sres/psy2way_{SEED}_{dd1}_{dd2}',
                           mating_regime=False,
                           metadata={'R_mate':R_mate,'h2':h2,
                            'dd1':dx[dd1], 'dd2':dx[dd2]})