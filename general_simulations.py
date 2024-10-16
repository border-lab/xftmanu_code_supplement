import dask
dask.config.set(scheduler='synchronous')

from xftsim.index import ComponentIndex
import xftsim as xft
import numpy as np
import pandas as pd
import dask.array as da
from dask.diagnostics import ProgressBar as PB
import time
import dask
import zarr
import argparse
import os.path

parser= argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=256000)
parser.add_argument('-m', type=int, default=4000)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--subsample', type=int, default=8000)
parser.add_argument('--kphen', type=int, default=5)
parser.add_argument('--kmate', type=int, default=5)
parser.add_argument('--rmate', type=float, default=0.2)
parser.add_argument('--theta', type=float, default=0.25)
parser.add_argument('--phi', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.0)
parser.add_argument('--h2', type=float, default=0.5)
parser.add_argument('--outpath', type=str, default='')
parser.add_argument('--env', type=bool, default=False)
args=parser.parse_args()

n = args.n
m = args.m
SEED  = args.seed
SSIZE = n
VSIZE = m
SUBSIZE=args.subsample
KMATE = args.kmate
KPHEN = args.kphen
θ = args.theta ##  vertical transmission
ϕ = args.phi ##  GxE
γ = args.gamma ## not used
h2 = args.h2 ## heritabilities
RMATE = args.rmate
OPATH = args.outpath 
ENV = args.env 
PREFIX = f'sres_{SEED}_{KMATE}_{KPHEN}_{RMATE}_{h2}_{θ}_{ϕ}_{γ}_{ENV}_{m}_pheno' 

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

reg_asrt = xft.mate.LinearAssortativeMatingRegime(offspring_per_pair=2, 
                                                  mates_per_female=1,
                                                  r=RMATE,
                                                  component_index=xft.index.ComponentIndex.from_product(phenotype_name = [f"y{k}" for k in range(KMATE)],
                                                                                                       component_name = 'phenotype'))

reg_rand = xft.mate.RandomMatingRegime(offspring_per_pair=2, 
                                       mates_per_female=2)
                                      

sample_stats =[xft.stats.SampleStatistics(), 
    xft.stats.HasemanElstonEstimator(filter_sample=False),
    xft.stats.MatingStatistics(),]


estimators = [
    xft.stats.HasemanElstonEstimatorSibship(filter_sample=False),
    xft.stats.Sib_GWAS_Estimator(n_sub=SUBSIZE, PGS=False, training_fraction=1.0),
    xft.stats.Pop_GWAS_Estimator(n_sub=SUBSIZE, PGS=False, training_fraction=1.0),
    xft.stats.Sib_GWAS_Estimator(n_sub=SUBSIZE/2, PGS=False, training_fraction=1.0,name='sib_GWAS_over_2'),
    xft.stats.Pop_GWAS_Estimator(n_sub=SUBSIZE/2, PGS=False, training_fraction=1.0,name='pop_GWAS_over_2'),
    xft.stats.Sib_GWAS_Estimator(n_sub=SUBSIZE/4, PGS=False, training_fraction=1.0,name='sib_GWAS_over_4'),
    xft.stats.Pop_GWAS_Estimator(n_sub=SUBSIZE/4, PGS=False, training_fraction=1.0,name='pop_GWAS_over_4'),
]
sim = xft.sim.Simulation(architecture=arch,
                       founder_haplotypes=founders,
                       mating_regime=reg_rand,
                       statistics=[],
                       # sample_filter=xft.index.RandomSiblingSubsampleFilter(k=SUBSIZE),
                       recombination_map=rmap,
                       filter_sample=False,
                       )


sim.mating_regime=reg_asrt



for GEN in range(6):
    sim.increment_generation()
    sim.reproduce()
    sim.compute_phenotypes()
    sim.statistics = estimators
    sim.estimate_statistics()
    if GEN > 0:
        sim.haplotypes = sim.haplotypes[::2,:]
        sim.phenotypes = sim.phenotypes[::2,:]
    sim.mate()
    sim.update_pedigree()
    sim.apply_filter()
    sim.statistics = sample_stats
    sim.estimate_statistics()
    sim.process()


####################### PROCESS RESULTS #######################


def R2(a,b):
    return np.corrcoef(a,b,rowvar=True)[0,1]**2


def R1(a,b):
    return np.corrcoef(a,b,rowvar=True)[0,1]

import scipy.stats as stats

pindexer = [(f'y{i}','phenotype', 'proband') for i in range(args.kphen)]
def parse_gen(GEN=0):
    print(args)
    print(GEN)
    res=dict(results_store=sim.results_store,
             architecture=sim.architecture.components[0].effects,
             mating_store=sim.mating_store,
             phenotype_store=sim.phenotype_store,
                   )


    OUTPUT = dict(seed = args.seed,
                  gen = GEN,
                  args_n = args.n, 
                  args_m = args.m, 
                  args_subsample = args.subsample, 
                  args_kphen = args.kphen, 
                  args_kmate = args.kmate, 
                  args_rmate = args.rmate,
                  args_theta = args.theta,
                  args_phi = args.phi,
                  args_gamma = args.gamma,
                  args_h2 = args.h2,
                  args_env = args.env,
                 )
    
    ## true VCs
    
    tmp = res['phenotype_store'][GEN]
    mate = res['mating_store'][GEN]
    mframe = mate.get_mating_frame()
    
    addGen = tmp.xft[{'component_name':'addGen','vorigin_relative':-1}][:,:args.kmate]
    noise = tmp.xft[{'component_name':'noise','vorigin_relative':-1}][:,:args.kmate]
    vert = tmp.xft[{'component_name':'vert','vorigin_relative':-1}][:,:args.kmate]
    env = tmp.xft[{'component_name':'env','vorigin_relative':-1}][:,:args.kmate]
    gxe = tmp.xft[{'component_name':'gxe','vorigin_relative':-1}][:,:args.kmate]
    tot = tmp.xft[{'component_name':'phenotype','vorigin_relative':-1}][:,:args.kmate]
    beta_v = np.sum(res['architecture'].beta_unscaled_standardized_diploid**2,axis=0)[:args.kmate]
    tmp = np.corrcoef(res['architecture'].beta_unscaled_standardized_diploid.T)
    lt5_inds = np.tril_indices_from(tmp,k=-1)
    OUTPUT['rbeta_true'] = np.mean(tmp[lt5_inds])
    
    ag = (addGen.to_numpy()).var(axis=0)
    agn = (addGen.to_numpy() + noise.to_numpy()).var(axis=0)
    age = (addGen.to_numpy() + env.to_numpy()).var(axis=0)
    agex = (addGen.to_numpy() + env.to_numpy() + gxe.to_numpy()).var(axis=0)
    tv = (tot.to_numpy()).var(axis=0)
    
    rg_true = np.corrcoef(addGen.T)
    OUTPUT['rg_true']=np.mean(rg_true[lt5_inds])
    OUTPUT['vbeta_true'] = np.mean(beta_v/tv)
    OUTPUT['vg_true'] = np.mean(ag/tv)
    OUTPUT['vge_true'] = np.mean(agn/tv)
    OUTPUT['vgev_true'] = np.mean(age/tv)
    OUTPUT['vgevx_true'] = np.mean(agex/tv)
    OUTPUT['vtot_full'] = np.mean(tot).values
    OUTPUT['h2_true'] = np.mean(ag/tv)

    ## HE
    OUTPUT['he_rg'] = np.mean(res['results_store'][GEN]['HE_regression']['corr_HE'].loc[pindexer,pindexer].values[lt5_inds])
    OUTPUT['he_sib_rg'] = np.mean(res['results_store'][GEN]['HE_regression_sibship']['corr_HEsib'].loc[pindexer,pindexer].values[lt5_inds])
    he_h2 = res['results_store'][GEN]['HE_regression']['cov_HE'].loc[pindexer,pindexer].to_numpy()
    he_sib_h2 = res['results_store'][GEN]['HE_regression_sibship']['cov_HEsib'].loc[pindexer,pindexer].to_numpy()
    OUTPUT['he_h2'] = np.mean(np.diag(he_h2))
    OUTPUT['he_sib_h2'] = np.mean(np.diag(he_sib_h2))
    ## GWAS 
    pgwas = res['results_store'][GEN]['pop_GWAS']
    p_est = pgwas['estimates'].loc[:,:,[f'y{i}.phenotype.proband' for i in range(args.kphen)]]
    beta_true = res['architecture'].beta_raw_diploid
    beta_hat = p_est[:,0,:]
    pval_pgwas = p_est[:,3,:]
    sgwas = res['results_store'][GEN]['sib_GWAS']
    s_est = sgwas['estimates'].loc[:,:,[f'y{i}.phenotype.proband' for i in range(args.kphen)]]
    sbeta_hat = s_est[:,0,:]
    pval_sgwas = s_est[:,3,:]

    OUTPUT['rbeta_hat_pgwas'] = np.mean(np.corrcoef(beta_hat.values.T)[lt5_inds])
    OUTPUT['rbeta_hat_sgwas'] = np.mean(np.corrcoef(sbeta_hat.values.T)[lt5_inds])
    
    gwas_hits = dict()
    args.m_causal = args.m // args.kphen #np.sum(beta_true!=0,axis=0)[0] 
    OUTPUT['args_m_causal'] = args.m_causal
    for threshold in [.05,.5,.05/args.m_causal, .5/args.m_causal]:
        gwas_hits[f'pgwas_true_positives_{threshold}'] = np.sum(np.logical_and(pval_pgwas<threshold, beta_true!=0)).data / np.sum(beta_true!=0).data
        gwas_hits[f'pgwas_false_positives_{threshold}'] = np.sum(np.logical_and(pval_pgwas<threshold, beta_true==0)).data / np.sum(beta_true==0).data
    
    OUTPUT.update(gwas_hits)
    snlog10p = -np.log10(stats.norm.cdf(-np.abs(s_est[:,2,:])))
    sgwas_hits = dict()
    for threshold in [.05,.5,.05/args.m_causal, .5/args.m_causal]:
        sgwas_hits[f'sgwas_true_positives_{threshold}'] = np.sum(np.logical_and(pval_sgwas<threshold, beta_true!=0)).data / np.sum(beta_true!=0).data
        sgwas_hits[f'sgwas_false_positives_{threshold}'] = np.sum(np.logical_and(pval_sgwas<threshold, beta_true==0)).data / np.sum(beta_true==0).data
    OUTPUT.update(sgwas_hits)
    
    ## GWAS 2
    pgwas = res['results_store'][GEN]['pop_GWAS_over_2']
    p_est = pgwas['estimates'].loc[:,:,[f'y{i}.phenotype.proband' for i in range(args.kphen)]]
    beta_true = res['architecture'].beta_raw_diploid
    beta_hat = p_est[:,0,:]
    pval_pgwas = p_est[:,3,:]
    sgwas = res['results_store'][GEN]['sib_GWAS_over_2']
    s_est = sgwas['estimates'].loc[:,:,[f'y{i}.phenotype.proband' for i in range(args.kphen)]]
    sbeta_hat = s_est[:,0,:]
    pval_sgwas = s_est[:,3,:]

    OUTPUT['rbeta_hat_pgwas_over_2'] = np.mean(np.corrcoef(beta_hat.values.T)[lt5_inds])
    OUTPUT['rbeta_hat_sgwas_over_2'] = np.mean(np.corrcoef(sbeta_hat.values.T)[lt5_inds])
    
    gwas_hits = dict()
    args.m_causal = np.sum(beta_true==0,axis=0)[0]
    OUTPUT['args_m_causal'] = args.m_causal
    for threshold in [.05,.5,.05/args.m_causal, .5/args.m_causal]:
        gwas_hits[f'pgwas_over_2_true_positives_{threshold}'] = np.mean(np.logical_and(pval_pgwas<threshold, beta_true!=0)).data
        gwas_hits[f'pgwas_over_2_false_positives_{threshold}'] = np.mean(np.logical_and(pval_pgwas<threshold, beta_true==0)).data
    
    OUTPUT.update(gwas_hits)
    snlog10p = -np.log10(stats.norm.cdf(-np.abs(s_est[:,2,:])))
    sgwas_hits = dict()
    for threshold in [.05,.5,.05/args.m_causal, .5/args.m_causal]:
        sgwas_hits[f'sgwas_over_2_true_positives_{threshold}'] = np.mean(np.logical_and(pval_sgwas<threshold, beta_true!=0)).data
        sgwas_hits[f'sgwas_over_2_false_positives_{threshold}'] = np.mean(np.logical_and(pval_sgwas<threshold, beta_true==0)).data
    OUTPUT.update(sgwas_hits)

 
    ## GWAS 4
    pgwas = res['results_store'][GEN]['pop_GWAS_over_4']
    p_est = pgwas['estimates'].loc[:,:,[f'y{i}.phenotype.proband' for i in range(args.kphen)]]
    beta_true = res['architecture'].beta_raw_diploid
    beta_hat = p_est[:,0,:]
    pval_pgwas = p_est[:,3,:]
    sgwas = res['results_store'][GEN]['sib_GWAS_over_4']
    s_est = sgwas['estimates'].loc[:,:,[f'y{i}.phenotype.proband' for i in range(args.kphen)]]
    sbeta_hat = s_est[:,0,:]
    pval_sgwas = s_est[:,3,:]

    OUTPUT['rbeta_hat_pgwas_over_4'] = np.mean(np.corrcoef(beta_hat.values.T)[lt5_inds])
    OUTPUT['rbeta_hat_sgwas_over_4'] = np.mean(np.corrcoef(sbeta_hat.values.T)[lt5_inds])
    
    gwas_hits = dict()
    args.m_causal = np.sum(beta_true==0,axis=0)[0]
    OUTPUT['args_m_causal'] = args.m_causal
    for threshold in [.05,.5,.05/args.m_causal, .5/args.m_causal]:
        gwas_hits[f'pgwas_over_4_true_positives_{threshold}'] = np.mean(np.logical_and(pval_pgwas<threshold, beta_true!=0)).data
        gwas_hits[f'pgwas_over_4_false_positives_{threshold}'] = np.mean(np.logical_and(pval_pgwas<threshold, beta_true==0)).data
    
    OUTPUT.update(gwas_hits)
    snlog10p = -np.log10(stats.norm.cdf(-np.abs(s_est[:,2,:])))
    sgwas_hits = dict()
    for threshold in [.05,.5,.05/args.m_causal, .5/args.m_causal]:
        sgwas_hits[f'sgwas_over_4_true_positives_{threshold}'] = np.mean(np.logical_and(pval_sgwas<threshold, beta_true!=0)).data
        sgwas_hits[f'sgwas_over_4_false_positives_{threshold}'] = np.mean(np.logical_and(pval_sgwas<threshold, beta_true==0)).data
    OUTPUT.update(sgwas_hits)

    ## mating
    mcors = res['results_store'][GEN]['mating_statistics']['mate_correlations'].stack()
    vnames = [f"rmate_{'_'.join(x)}" for x in mcors.index]
    for vname,mcor in zip(vnames,mcors.values):
        OUTPUT[vname] = mcor
        
    return OUTPUT

oo = [parse_gen(gen) for gen in range(6)]
output = pd.concat([pd.DataFrame(oo[gen], index=[f"{oo[gen]['seed']}_{gen}_{oo[gen]['args_kmate']}_{oo[gen]['args_theta']}_"
                                                 f"{oo[gen]['args_phi']}_{oo[gen]['args_env']}"]) for gen in range(6)])

output.to_csv(f"{OPATH}/{PREFIX}_parsed.csv")
 
## save phenotypes
for gen in range(6):
    sim.phenotype_store[gen].to_pandas().to_csv(f"{OPATH}/{PREFIX}_phenos{gen}.csv")