import os
NUMTHREAD = str(1)
os.environ["NUMBA_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = NUMTHREAD
os.environ["OPENBLAS_NUM_THREADS"] = NUMTHREAD
os.environ["MKL_NUM_THREADS"] = NUMTHREAD
os.environ["VECLIB_MAXIMUM_THREADS"] = NUMTHREAD
os.environ["NUMEXPR_NUM_THREADS"] = NUMTHREAD

from dask import config
config.set(scheduler='synchronous')
import dask

from xftsim.index import ComponentIndex
import xftsim as xft
import numpy as np
import pandas as pd
import dask.array as da
# import allel
from dask.diagnostics import ProgressBar as PB
import time
import dask
import zarr
import argparse
import os.path

parser= argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=32000)
parser.add_argument('-m', type=int, default=1000)

parser.add_argument('--seed', type=int, default=11)
parser.add_argument('--subsample', type=int, default=8000)
parser.add_argument('--kphen', type=int, default=2)
parser.add_argument('--kmate', type=int, default=2)
parser.add_argument('--rmate', type=float, default=0.5)
parser.add_argument('--theta', type=float, default=0.)
parser.add_argument('--phi', type=float, default=0.)
parser.add_argument('--gamma', type=float, default=0.0)
parser.add_argument('--h2', type=float, default=0.5)
parser.add_argument('--outpath' , type=str, default='~/data/ribo7412/ascertainment_sims062424/')
parser.add_argument('--env', type=bool, default=False)
args=parser.parse_args()


n = args.n
m = args.m * args.kphen
SEED  = args.seed
SSIZE = n
VSIZE = m
SUBSIZE=args.subsample
KMATE = args.kmate
KPHEN = args.kphen
θ = args.theta ## fraction vertical transmission
ϕ = args.phi ## fraction GxE
γ = args.gamma ## fraction causal interdependence
h2 = args.h2 ## heritabilities
RMATE = args.rmate
OPATH = args.outpath 
ENV = args.env 
PREFIX = f'ascertainment_{SEED}_{KMATE}_{KPHEN}_{RMATE}_{h2}_{θ}_{ϕ}_{γ}_{ENV}_{m}_pheno' 
args.prefix = PREFIX
founders = xft.founders.founder_haplotypes_uniform_AFs(n=n,m=m,minMAF=.05)
rmap =xft.reproduce.RecombinationMap.constant_map_from_haplotypes(founders, p = 50/m)

####################### SETUP SIM #######################
ve=1-h2
vg=np.repeat(h2,KPHEN)
ve=1-vg
phenos = xft.utils.paste(['y',np.arange(KPHEN)], sep='')

pheno_ind = ComponentIndex.from_product(phenos,'phenotype')
parent_pheno_ind = ComponentIndex.from_product(phenos,'phenotype',[0,1])
parent_env_ind = ComponentIndex.from_product(phenos,'env',[0,1])
parent_pheno_ind.comp_type = 'outcome'
intermediate_ind = ComponentIndex.from_product(phenos,'intermediate')
genetic_ind = ComponentIndex.from_product(phenos,'addGen')
inherited_ind = ComponentIndex.from_product(phenos,'vert')
noise_ind = ComponentIndex.from_product(phenos,'noise')
dependent_ind = ComponentIndex.from_product(phenos,'dependent')
env_ind = ComponentIndex.from_product(phenos,'env')
gxe_ind = ComponentIndex.from_product(phenos,'gxe')


np.random.seed(SEED)
eff = xft.effect.NonOverlappingEffects(vg = np.ones(KPHEN)*h2,
                                      variant_indexer=founders.xft.get_variant_indexer(),
                                      component_indexer=genetic_ind)

genetic_comp = xft.arch.AdditiveGeneticComponent(eff)
noise_comp = xft.arch.AdditiveNoiseComponent(variances=ve-θ-ϕ-γ,
                                             component_index=noise_ind)
if not ENV:
    vert_comp = xft.arch.LinearVerticalComponent(input_cindex = parent_pheno_ind,
                                             output_cindex= inherited_ind,
                                             coefficient_matrix=np.ones((KPHEN,2*KPHEN))*np.sqrt(θ / (2*KPHEN)),
                                             founder_variances=np.ones(KPHEN*2),
                                             normalize=True)
else:
    vert_comp = xft.arch.LinearVerticalComponent(input_cindex = parent_env_ind,
                                             output_cindex= inherited_ind,
                                             coefficient_matrix=np.ones((KPHEN,2*KPHEN))*np.sqrt(θ/ve/(2*KPHEN)),
                                             founder_variances=np.ones(KPHEN*2)*ve,
                                             normalize=True)


### FOUNDER variances broken

env_comp = xft.arch.SumAllTransformation(input_cindex=inherited_ind.merge(noise_ind), output_component_name='env', output_comp_type='intermediate')

gxe_comps = [xft.arch.ProductComponent(input_cindex=ComponentIndex.from_product(pheno,['env','addGen']),
                                       output_cindex=ComponentIndex.from_product(pheno,['gxe']),
                                       normalize=False, 
                                       output_coef = np.sqrt(ϕ/(vg[i]*(ve[i]-θ-γ))),
                                       ) for i,pheno in enumerate(phenos)]
inter_comp = xft.arch.SumAllTransformation(input_cindex=env_ind.merge(genetic_ind).merge(gxe_ind),
                                           output_component_name='phenotype',
                                           output_comp_type='outcome')


arch = xft.arch.Architecture([genetic_comp, noise_comp, vert_comp, env_comp] + gxe_comps + [inter_comp])

reg_asrt = xft.mate.FilteredMatingRegime(
                                         xft.mate.LinearAssortativeMatingRegime(offspring_per_pair=2, 
                                                  mates_per_female=6,
                                                  r=RMATE,
                                                  component_index=xft.index.ComponentIndex.from_product(phenotype_name = [f"y{k}" for k in range(KMATE)],
                                                                                                       component_name = 'phenotype')),
                                         xft.filters.UnrelatedSampleFilter(nsub=n))

reg_rand = xft.mate.FilteredMatingRegime(xft.mate.RandomMatingRegime(offspring_per_pair=2,
                                                            mates_per_female=6),
    xft.filters.UnrelatedSampleFilter(nsub=n))

                                      

sample_stats =[xft.stats.SampleStatistics(), 
    # xft.stats.HasemanElstonEstimator(filter_sample=False),
    xft.stats.MatingStatistics(),]

# estimators = [
#     xft.stats.Sib_GWAS_Estimator(n_sub=SUBSIZE, PGS_sub_divisions=1),
#     xft.stats.Pop_GWAS_Estimator(n_sub=SUBSIZE, PGS_sub_divisions=1),
# ]

COEF=np.concatenate([[1], np.zeros(KPHEN-1)])
COEF=np.ones(KPHEN)
COMBINE='max'



pop_unasc_filter = xft.filters.UnrelatedSampleFilter(nsub=SUBSIZE)
pop_asc_filter_top2 = xft.filters.UnrelatedAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=SUBSIZE*2, coef=COEF)#,
pop_asc_filter_top4 = xft.filters.UnrelatedAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=SUBSIZE*4, coef=COEF)#,
pop_asc_filter_bottom4 = xft.filters.UnrelatedAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=SUBSIZE*4, coef=-COEF)#,
pop_asc_filter_bottom2 = xft.filters.UnrelatedAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=SUBSIZE*2, coef=-COEF)#,

sib_unasc_filter = xft.filters.SibpairSampleFilter(nsub=SUBSIZE)
sib_asc_filter_max_top2   = xft.filters.SibpairAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=int(SUBSIZE*2.0), coef=COEF, combine='max')
sib_asc_filter_min_top2   = xft.filters.SibpairAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=int(SUBSIZE*2.0), coef=COEF, combine='min')
sib_asc_filter_mean_top2   = xft.filters.SibpairAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=int(SUBSIZE*2.0), coef=COEF, combine='mean')
sib_asc_filter_max_bottom2   = xft.filters.SibpairAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=int(SUBSIZE*2.0), coef=-COEF, combine='max')
sib_asc_filter_min_bottom2   = xft.filters.SibpairAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=int(SUBSIZE*2.0), coef=-COEF, combine='min')
sib_asc_filter_mean_bottom2   = xft.filters.SibpairAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=int(SUBSIZE*2.0), coef=-COEF, combine='mean')
sib_asc_filter_max_top4   = xft.filters.SibpairAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=int(SUBSIZE*4.0), coef=COEF, combine='max')
sib_asc_filter_min_top4   = xft.filters.SibpairAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=int(SUBSIZE*4.0), coef=COEF, combine='min')
sib_asc_filter_mean_top4   = xft.filters.SibpairAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=int(SUBSIZE*4.0), coef=COEF, combine='mean')
sib_asc_filter_max_bottom4   = xft.filters.SibpairAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=int(SUBSIZE*4.0), coef=-COEF, combine='max')
sib_asc_filter_min_bottom4   = xft.filters.SibpairAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=int(SUBSIZE*4.0), coef=-COEF, combine='min')
sib_asc_filter_mean_bottom4   = xft.filters.SibpairAscertainmentFilter(nsub_ascertained=SUBSIZE,nsub_random=int(SUBSIZE*4.0), coef=-COEF, combine='mean')

cind = xft.index.ComponentIndex.from_product(['y'+str(i) for i in range(KMATE)],['addGen','phenotype'])

estimators = [

### GWAS
    xft.stats.Pop_GWAS_Estimator(name='pop_GWAS_unasc', PGS=False, training_fraction=1.0,sample_filter=pop_unasc_filter),
    xft.stats.Pop_GWAS_Estimator(name='pop_GWAS_asc_top2', PGS=False, training_fraction=1.0,sample_filter=pop_asc_filter_top2),
    xft.stats.Pop_GWAS_Estimator(name='pop_GWAS_asc_top4', PGS=False, training_fraction=1.0,sample_filter=pop_asc_filter_top4),
    xft.stats.Pop_GWAS_Estimator(name='pop_GWAS_asc_bottom4', PGS=False, training_fraction=1.0,sample_filter=pop_asc_filter_bottom4),
    xft.stats.Pop_GWAS_Estimator(name='pop_GWAS_asc_bottom2', PGS=False, training_fraction=1.0,sample_filter=pop_asc_filter_bottom2),

    xft.stats.Sib_GWAS_Estimator(name='sib_GWAS_unasc', PGS=False, training_fraction=1.0,sample_filter=sib_unasc_filter),
    xft.stats.Sib_GWAS_Estimator(name='sib_GWAS_asc_max_top2', PGS=False, training_fraction=1.0,sample_filter=sib_asc_filter_max_top2),
    xft.stats.Sib_GWAS_Estimator(name='sib_GWAS_asc_min_top2', PGS=False, training_fraction=1.0,sample_filter=sib_asc_filter_min_top2),
    xft.stats.Sib_GWAS_Estimator(name='sib_GWAS_asc_mean_top2', PGS=False, training_fraction=1.0,sample_filter=sib_asc_filter_mean_top2),
    xft.stats.Sib_GWAS_Estimator(name='sib_GWAS_asc_max_bottom2', PGS=False, training_fraction=1.0,sample_filter=sib_asc_filter_max_bottom2),
    xft.stats.Sib_GWAS_Estimator(name='sib_GWAS_asc_min_bottom2', PGS=False, training_fraction=1.0,sample_filter=sib_asc_filter_min_bottom2),
    xft.stats.Sib_GWAS_Estimator(name='sib_GWAS_asc_mean_bottom2', PGS=False, training_fraction=1.0,sample_filter=sib_asc_filter_mean_bottom2),
    xft.stats.Sib_GWAS_Estimator(name='sib_GWAS_asc_max_top4', PGS=False, training_fraction=1.0,sample_filter=sib_asc_filter_max_top4),
    xft.stats.Sib_GWAS_Estimator(name='sib_GWAS_asc_min_top4', PGS=False, training_fraction=1.0,sample_filter=sib_asc_filter_min_top4),
    xft.stats.Sib_GWAS_Estimator(name='sib_GWAS_asc_mean_top4', PGS=False, training_fraction=1.0,sample_filter=sib_asc_filter_mean_top4),
    xft.stats.Sib_GWAS_Estimator(name='sib_GWAS_asc_max_bottom4', PGS=False, training_fraction=1.0,sample_filter=sib_asc_filter_max_bottom4),
    xft.stats.Sib_GWAS_Estimator(name='sib_GWAS_asc_min_bottom4', PGS=False, training_fraction=1.0,sample_filter=sib_asc_filter_min_bottom4),
    xft.stats.Sib_GWAS_Estimator(name='sib_GWAS_asc_mean_bottom4', PGS=False, training_fraction=1.0,sample_filter=sib_asc_filter_mean_bottom4),


### HE
    xft.stats.HasemanElstonEstimator(name='pop_HE_unasc',sample_filter=pop_unasc_filter),
    xft.stats.HasemanElstonEstimator(name='pop_HE_asc_top2',sample_filter=pop_asc_filter_top2),
    xft.stats.HasemanElstonEstimator(name='pop_HE_asc_top4',sample_filter=pop_asc_filter_top4),
    xft.stats.HasemanElstonEstimator(name='pop_HE_asc_bottom4',sample_filter=pop_asc_filter_bottom4),
    xft.stats.HasemanElstonEstimator(name='pop_HE_asc_bottom2',sample_filter=pop_asc_filter_bottom2),
    xft.stats.HasemanElstonEstimatorSibship(name='sib_HE_unasc',sample_filter=sib_unasc_filter),
    xft.stats.HasemanElstonEstimatorSibship(name='sib_HE_asc_max_top2',sample_filter=sib_asc_filter_max_top2),
    xft.stats.HasemanElstonEstimatorSibship(name='sib_HE_asc_min_top2',sample_filter=sib_asc_filter_min_top2),
    xft.stats.HasemanElstonEstimatorSibship(name='sib_HE_asc_mean_top2',sample_filter=sib_asc_filter_mean_top2),
    xft.stats.HasemanElstonEstimatorSibship(name='sib_HE_asc_max_bottom2',sample_filter=sib_asc_filter_max_bottom2),
    xft.stats.HasemanElstonEstimatorSibship(name='sib_HE_asc_min_bottom2',sample_filter=sib_asc_filter_min_bottom2),
    xft.stats.HasemanElstonEstimatorSibship(name='sib_HE_asc_mean_bottom2',sample_filter=sib_asc_filter_mean_bottom2),
    xft.stats.HasemanElstonEstimatorSibship(name='sib_HE_asc_max_top4',sample_filter=sib_asc_filter_max_top4),
    xft.stats.HasemanElstonEstimatorSibship(name='sib_HE_asc_min_top4',sample_filter=sib_asc_filter_min_top4),
    xft.stats.HasemanElstonEstimatorSibship(name='sib_HE_asc_mean_top4',sample_filter=sib_asc_filter_mean_top4),
    xft.stats.HasemanElstonEstimatorSibship(name='sib_HE_asc_max_bottom4',sample_filter=sib_asc_filter_max_bottom4),
    xft.stats.HasemanElstonEstimatorSibship(name='sib_HE_asc_min_bottom4',sample_filter=sib_asc_filter_min_bottom4),
    xft.stats.HasemanElstonEstimatorSibship(name='sib_HE_asc_mean_bottom4',sample_filter=sib_asc_filter_mean_bottom4),


### Sample stats
    xft.stats.SampleStatistics(name='pop_sample_unasc',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=pop_unasc_filter),
    xft.stats.SampleStatistics(name='pop_sample_asc_top2',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=pop_asc_filter_top2),
    xft.stats.SampleStatistics(name='pop_sample_asc_top4',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=pop_asc_filter_top4),
    xft.stats.SampleStatistics(name='pop_sample_asc_bottom4',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=pop_asc_filter_bottom4),
    xft.stats.SampleStatistics(name='pop_sample_asc_bottom2',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=pop_asc_filter_bottom2),
    xft.stats.SampleStatistics(name='sib_sample_unasc',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=sib_unasc_filter),
    xft.stats.SampleStatistics(name='sib_sample_asc_max_top2',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=sib_asc_filter_max_top2),
    xft.stats.SampleStatistics(name='sib_sample_asc_min_top2',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=sib_asc_filter_min_top2),
    xft.stats.SampleStatistics(name='sib_sample_asc_mean_top2',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=sib_asc_filter_mean_top2),
    xft.stats.SampleStatistics(name='sib_sample_asc_max_bottom2',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=sib_asc_filter_max_bottom2),
    xft.stats.SampleStatistics(name='sib_sample_asc_min_bottom2',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=sib_asc_filter_min_bottom2),
    xft.stats.SampleStatistics(name='sib_sample_asc_mean_bottom2',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=sib_asc_filter_mean_bottom2),
    xft.stats.SampleStatistics(name='sib_sample_asc_max_top4',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=sib_asc_filter_max_top4),
    xft.stats.SampleStatistics(name='sib_sample_asc_min_top4',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=sib_asc_filter_min_top4),
    xft.stats.SampleStatistics(name='sib_sample_asc_mean_top4',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=sib_asc_filter_mean_top4),
    xft.stats.SampleStatistics(name='sib_sample_asc_max_bottom4',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=sib_asc_filter_max_bottom4),
    xft.stats.SampleStatistics(name='sib_sample_asc_min_bottom4',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=sib_asc_filter_min_bottom4),
    xft.stats.SampleStatistics(name='sib_sample_asc_mean_bottom4',component_index=cind,variances=False, vcov=True, corr=False,sample_filter=sib_asc_filter_mean_bottom4),
    ]




sim = xft.sim.Simulation(architecture=arch,
                       founder_haplotypes=founders,
                       mating_regime=reg_rand,
                       statistics=[],
                       recombination_map=rmap,
                       filter_sample=False,
                       )
sim.run(1)
sim.mating_regime=reg_asrt

sim.run(4)
sim.statistics = estimators
sim.run(1)

ALPHA=.05
eff = sim.architecture.components[0].effects.beta_scaled_standardized_diploid
cv = eff!=0


output = vars(args).copy()
sample_output = vars(args).copy()

est_names = [x.name for x in estimators]

gwas_estimators = [x for x in est_names if x.rfind('GWAS')>=0]
he_estimators = [x for x in est_names if x.rfind('HE')>=0]
sample_estimators = [x for x in est_names if x.rfind('sample')>=0]


def process_multi_ind_dict(dd, prefix=''):
    return {(prefix+'.'.join(key)):value for key,value in dd.items()}


for ss in sample_estimators:
    res = sim.results[ss]['means'].to_dict()
    sample_output.update(process_multi_ind_dict(res,'means_'+ss+'_'))
    res = sim.results[ss]['vcov'].to_dict()
    for key,x in process_multi_ind_dict(res).items():
        sample_output.update(process_multi_ind_dict(x,'vcov_' + ss+'_'+key+'_') )

for ss in gwas_estimators:
    pp = sim.results[ss]['estimates'][:,3,:KMATE]
    bb = sim.results[ss]['estimates'][:,0,:KMATE]
    le5 = pp.values[:,:KMATE]<=ALPHA
    tpr = np.logical_and(cv[:,:KMATE],le5).sum(0) / cv[:,:KMATE].sum(0)
    fpr = np.logical_and(~cv[:,:KMATE],le5).sum(0) / (~cv)[:,:KMATE].sum(0)
    betacorr = bb.to_pandas().corr().values
    output.update({('gwas_fpr_'+ss):fpr.mean(),
                   ('gwas_tpr_'+ss):tpr.mean(),
                   ('gwas_rbeta_'+ss):betacorr[np.tril_indices_from(betacorr,-1)].mean(),
                   })

for ss in he_estimators:
    hecov = sim.results[ss]['cov_HE']
    hecorr = sim.results[ss]['corr_HE']
    h2 = np.diag(hecov.values)[:KMATE].mean()
    rg = hecorr.values[:KMATE,:KMATE]
    rg = rg[np.tril_indices_from(rg,-1)].mean()
    output.update({('he_h2_'+ss):h2,('he_rg_'+ss):rg,})

pd.DataFrame.from_records(sample_output, index=[0]).to_csv(f"{OPATH}/{PREFIX}_sample_stats.csv")
pd.DataFrame.from_records(output, index=[0]).to_csv(f"{OPATH}/{PREFIX}_estimators.csv")

"""
parallel -j 96 --delay 2s --retries 1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python3 ascertainment_sims.py --outpath ~/data/ribo7412/ascertainment_sims062524/ \
    -m 800 --phi {5} --theta {2} --kmate {3} --kphen {3} --rmate {4} --seed {1} \
    ::: {1..500} ::: 0 .05 .2  ::: 2 5 ::: 0 .2 ::: 0 .05 .2
"""