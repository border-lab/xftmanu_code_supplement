import os

NUMTHREAD = '16'
NUMTHREAD_NUMBA = '1'
NUMTHREAD_LS = '8'
os.environ["OMP_NUM_THREADS"] = NUMTHREAD
os.environ["OPENBLAS_NUM_THREADS"] = NUMTHREAD 
os.environ["MKL_NUM_THREADS"] = NUMTHREAD
os.environ["VECLIB_MAXIMUM_THREADS"] = NUMTHREAD 
os.environ["NUMEXPR_NUM_THREADS"] = NUMTHREAD 
os.environ["NUMBA_NUM_THREADS"] = NUMTHREAD_NUMBA

import xftsim as xft
import numpy as np
import pandas as pd
import scipy.stats as stats
TIMELIMIT=60
MATEBATCH=500
THREADS=2 
SUBSIZE=4000
import argparse

parser= argparse.ArgumentParser()
parser.add_argument('-m', type=int, default=2000)
parser.add_argument('-n', type=int, default=128000)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--path', type=str, default='/data1/ribo7412/edu_no_CD_LS.01/')
parser.add_argument('--h2', type=float, default=.01)
args=parser.parse_args()



OPATH=args.path
SEED=args.seed
H2=args.h2

n=args.n
m=args.m


PREFIX=f'edu_{SEED}_{m}_{H2}'
print(f'SEED = {SEED}')
h2=np.array([H2, .6])

np.random.seed(SEED)
cind = xft.index.ComponentIndex.from_product(['edu','height','wealth'], 
                                             'phenotype')

xmatecorr = np.array([[0.455587545433721, 0.124436818044673,  .2],
                      [0.124436818044673, 0.225952731137808,  .1],
                      [.2,.1,.3]])
xmatecorr = np.ones_like(xmatecorr)*np.mean(xmatecorr)
xmatecorr += np.diag(1-np.diag(xmatecorr))
xmatecorr = np.array([[0.24626319, 0.1915851 , 0.2521529 ],
       [0.1915851 , 0.1247212 , 0.18323772],
       [0.2521529 , 0.18323772, 0.25072489]])

mate_reg = xft.mate.GeneralAssortativeMatingRegime(offspring_per_pair=2, 
                                                   mates_per_female=1,
                                                   component_index=cind,
    cross_corr=xmatecorr,
    control=dict(nb_threads=int(NUMTHREAD_LS), tolerance = .001,
                 time_limit = 30,
                 time_between_displays=5))

#mate_reg = xft.mate.LinearAssortativeMatingRegime(offspring_per_pair=2, mates_per_female=1,
# component_index=cind,
#    r= np.mean(xmatecorr),)



bmate_reg = xft.mate.BatchedMatingRegime(mate_reg, max_batch_size=MATEBATCH)
mate_batch_size =MATEBATCH
# bmate_reg = mate_reg

####################### SETUP SIM #######################


founders = xft.founders.founder_haplotypes_uniform_AFs(n=n,m=m,minMAF=.05)
rmap =xft.reproduce.RecombinationMap.constant_map_from_haplotypes(founders, p = 50/m)

gen_ind = xft.index.ComponentIndex.from_product( ['edu','height'], ['addGen'])
noise_ind = xft.index.ComponentIndex.from_product( ['edu','height','wealth'], ['noise'])
vert_ind = xft.index.ComponentIndex.from_product( ['edu','height','wealth'], ['vert'])
intermediate_ind = xft.index.ComponentIndex.from_product(['edu','height','wealth'],'intermediate')
pheno_ind = xft.index.ComponentIndex.from_product(['edu','height','wealth'],'phenotype')
vert_input = xft.index.ComponentIndex.from_product(['edu','wealth'], ['phenotype'], [0,1])

# BB = 0.995266
# AA = -0.495266
# additive phenogenetic architecture
gen_comp = xft.arch.AdditiveGeneticComponent(
    xft.effect.GCTAEffects(vg=h2, 
                           variant_indexer= founders.xft.get_variant_indexer(),
                           component_indexer= gen_ind))

noise_comp = xft.arch.AdditiveNoiseComponent(
    variances= np.array([1/3-H2, .4, 1/3]),
    component_index= noise_ind)


founder_variances = np.sqrt([1,1,1,1])
# founder_variances = np.sqrt([1,1,1,1])

CC = np.sqrt(1/3)
AA = np.sqrt(2/3)

##                              edu_m edu_f  wl_m  wl_f  
transmission_matrix =np.array([[   CC,   CC,   CC,   CC],               # edu_vert
                               [  0.0,  0.0,  0.0,  0.0],               # hgt_vert
                               [  0.0,  0.0,   AA,   AA]])*np.sqrt(.5)  # wlt_vert

vtcomp = xft.arch.LinearVerticalComponent(input_cindex=vert_input,
                                          output_cindex=vert_ind,
                                          founder_variances=founder_variances,
                                          coefficient_matrix=transmission_matrix,
                                          normalize = True)

sum_inds = noise_ind.merge(gen_ind).merge(vert_ind)


# ##                       ed_n  ht_n  wl_n  ht_g  ed_v  ht_v  wl_v
# smat = np.sqrt(np.array([[1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0],   # edu_inter
#                          [0.0,  1.0,  0.0,  1.0,  0.0,  0.0,  0.0],   # hgt_inter
#                          [0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0]])) # wlt_inter


# inter_comp = xft.arch.LinearTransformationComponent(input_cindex=sum_inds,
#                                                     output_cindex=intermediate_ind,
#                                                     coefficient_matrix=smat,
#                                                     normalize=False)


inter_comp = xft.arch.SumAllTransformation(input_cindex=sum_inds,
                                           output_component_name='phenotype',
                                           output_comp_type='phenotype')

arch = xft.arch.Architecture([gen_comp, noise_comp, vtcomp, inter_comp])




sample_stats =[xft.stats.SampleStatistics(), 
    xft.stats.HasemanElstonEstimator(filter_sample=False,
                                     component_index = pheno_ind),
    xft.stats.MatingStatistics(),]
estimators = [
    xft.stats.HasemanElstonEstimatorSibship(filter_sample=False),
    xft.stats.Sib_GWAS_Estimator(n_sub=SUBSIZE, PGS=False, training_fraction=1.0),
    xft.stats.Pop_GWAS_Estimator(n_sub=SUBSIZE, PGS=False, training_fraction=1.0),
    xft.stats.Sib_GWAS_Estimator(n_sub=SUBSIZE//2, PGS=False, name='sgwas2', training_fraction=1.0),
    xft.stats.Pop_GWAS_Estimator(n_sub=SUBSIZE//2, PGS=False, name='pgwas2', training_fraction=1.0),
    xft.stats.Sib_GWAS_Estimator(n_sub=SUBSIZE//4, PGS=False, name='sgwas4', training_fraction=1.0),
    xft.stats.Pop_GWAS_Estimator(n_sub=SUBSIZE//4, PGS=False, name='pgwas4', training_fraction=1.0),
]

sim = xft.sim.Simulation(architecture=arch,
                           founder_haplotypes=founders,
                           mating_regime=bmate_reg,
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
    # sim.phenotype_store[GEN].to_pandas().to_csv(f"{OPATH}/{PREFIX}_phenos{GEN}.csv")
    # sim.haplotype_store[GEN].to_pandas().to_csv(f"{OPATH}/{PREFIX}_haplotypes{GEN}.csv")
    # print('write data')
    if GEN > 0:
        sim.haplotypes = sim.haplotypes[::2,:]
        sim.phenotypes = sim.phenotypes[::2,:]
    sim.mate()
    sim.update_pedigree()
    sim.apply_filter()
    sim.statistics = sample_stats
    sim.estimate_statistics()
    sim.process()



names = lambda x: '.'.join(x.split('.')[:2])

def cov2dict(x, prefix, vorigin=False, diag=True):
    tmp = x.stack(dropna=False)
    sorted2 = ['_'.join(sorted(list(x))) for x in tmp.index]
    dups = pd.Series(sorted2).duplicated()
    if not diag:
        for i,(a,b) in enumerate(tmp.index):
            if a==b:
                dups[i] = True 
    if vorigin:
        return {prefix+'_'+'_'.join([a,b]):c for (a,b),c in zip(tmp[~dups.values].index, 
                                                            tmp[~dups.values].values)}
    else:
        return {prefix+'_'+'_'.join([names(a), names(b)]):c for (a,b),c in zip(tmp[~dups.values].index, 
                                                                           tmp[~dups.values].values)}

def cov2dict2(x, prefix, diag=True, vorigin=False):
    tmp = x.copy()
    tmp.index=['.'.join(z) for z in tmp.index]
    tmp.columns=['.'.join(z) for z in tmp.columns]
    return cov2dict(tmp, prefix=prefix, diag=diag, vorigin=vorigin)

def R2(a,b):
    return np.corrcoef(a,b,rowvar=True)[0,1]**2


def R1(a,b):
    return np.corrcoef(a,b,rowvar=True)[0,1]

import scipy.stats as stats
BB=0
pindexer = [(f'y{i}','phenotype', 'proband') for i in range(5)]
def parse_gen(GEN=4):
    print(GEN)
    res=dict(results_store=sim.results_store,
             architecture=sim.architecture.components[0].effects,
             mating_store=sim.mating_store,
             phenotype_store=sim.phenotype_store,
                   )


    OUTPUT = dict(seed = SEED,
                  gen = GEN,
                  args_n = n, 
                  args_m = m, 
                  args_subsample = SUBSIZE, 
                  AA=AA,
                  BB=BB,
                  CC=CC,
                 )
    
    ## true VCs
    
    tmp = res['phenotype_store'][GEN]
    mate = res['mating_store'][GEN]
    mframe = mate.get_mating_frame()
    
    addGen = tmp.xft[{'component_name':'addGen','vorigin_relative':-1}]
    noise = tmp.xft[{'component_name':'noise','vorigin_relative':-1}]
    vert = tmp.xft[{'component_name':'vert','vorigin_relative':-1}]
    tot = tmp.xft[{'component_name':'phenotype','vorigin_relative':-1}]
    beta_v = np.sum(res['architecture'].beta_unscaled_standardized_diploid**2,axis=0)[0]
    lt3_inds = np.tril_indices(3,k=-1)
    
    pheno_wealth = tot.xft[dict(phenotype_name='wealth')].values
    pheno_edu = tot.xft[dict(phenotype_name='edu')].values
    pheno_height = tot.xft[dict(phenotype_name='height')].values
    addGen_edu = addGen.xft[dict(phenotype_name='edu')].values
    addGen_height = addGen.xft[dict(phenotype_name='height')].values
    addGen_total = addGen_edu + addGen_height

    OUTPUT['r2_true_edu.pheno_edu.addGen']=np.corrcoef(pheno_edu.T,addGen_edu.T)[0,1]**2
    OUTPUT['r2_true_edu.pheno_height.addGen']=np.corrcoef(pheno_edu.T,addGen_height.T)[0,1]**2
    OUTPUT['r2_true_edu.pheno_total.addGen']=np.corrcoef(pheno_edu.T,addGen_total.T)[0,1]**2

    OUTPUT['r2_true_wealth.pheno_edu.addGen']=np.corrcoef(pheno_wealth.T,addGen_edu.T)[0,1]**2
    OUTPUT['r2_true_wealth.pheno_height.addGen']=np.corrcoef(pheno_wealth.T,addGen_height.T)[0,1]**2
    OUTPUT['r2_true_wealth.pheno_total.addGen']=np.corrcoef(pheno_wealth.T,addGen_total.T)[0,1]**2

    OUTPUT['r2_true_height.pheno_edu.addGen']=np.corrcoef(pheno_height.T,addGen_edu.T)[0,1]**2
    OUTPUT['r2_true_height.pheno_height.addGen']=np.corrcoef(pheno_height.T,addGen_height.T)[0,1]**2
    OUTPUT['r2_true_height.pheno_total.addGen']=np.corrcoef(pheno_height.T,addGen_total.T)[0,1]**2
    
    ag = (addGen.to_numpy()).var(axis=0)
    tv = (tot.to_numpy()).var(axis=0)[:2]
    
    OUTPUT['h2_true_edu'] = (ag/tv)[0]
    OUTPUT['h2_true_height'] = (ag/tv)[1]
    OUTPUT['h2_true_wealth'] = 0.

    OUTPUT['vbeta_true_edu'] = 0.
    OUTPUT['vbeta_true_height'] = np.mean(beta_v/tv)
    OUTPUT['vbeta_true_wealth'] =0.
    
    OUTPUT['rg_true_edu_height'] = 0.
    OUTPUT['rg_true_edu_wealth'] = 1.
    OUTPUT['rg_true_height_wealth'] = 0.

    OUTPUT.update(cov2dict(tmp.xft[{'vorigin_relative':-1}].to_pandas().cov(), 'cov', vorigin=False))
    OUTPUT.update(cov2dict(tmp.xft[{'vorigin_relative':-1}].to_pandas().corr(), 'corr', vorigin=False, diag=False))


    ## HE
    phenos =['edu','height','wealth']
    OUTPUT.update(cov2dict2(res['results_store'][GEN]['HE_regression']['corr_HE'],'rgHE', diag=False, vorigin=True))
    he_h2 = np.diag(res['results_store'][GEN]['HE_regression']['cov_HE'])
    OUTPUT.update({f'h2HE_{x}':y for x,y in zip(phenos,he_h2)})
    OUTPUT.update(cov2dict2(res['results_store'][GEN]['HE_regression_sibship']['corr_HEsib'],'rgHEsib', diag=False, vorigin=True))
    he_h2 = np.diag(res['results_store'][GEN]['HE_regression_sibship']['cov_HEsib'])
    OUTPUT.update({f'h2HEsib_{x}':y for x,y in zip(phenos,he_h2)})
      
    ## GWAS 
    pgwas = res['results_store'][GEN]['pop_GWAS']
    p_est = pgwas['estimates']#.loc[:,:,[f'y{i}.phenotype.proband' for i in range(5)]]
    beta_true = res['architecture'].beta_raw_diploid
    beta_hat = p_est[:,0,:]
    beta_true = np.hstack([np.zeros_like(beta_true),beta_true,np.zeros_like(beta_true)])
    sgwas = res['results_store'][GEN]['sib_GWAS']
    s_est = sgwas['estimates']#.loc[:,:,[f'y{i}.phenotype.proband' for i in range(5)]]
    sbeta_hat = s_est[:,0,:]
    nlog10p = -np.log10(stats.norm.cdf(-np.abs(p_est[:,2,:])))

    OUTPUT.update(cov2dict(beta_hat.to_pandas().corr(), 'rbeta_hat_pgwas', diag=False))
    OUTPUT.update(cov2dict(sbeta_hat.to_pandas().corr(), 'rbeta_hat_sgwas', diag=False))

    ## GWAS 2
    pgwas = res['results_store'][GEN]['pgwas2']
    p_est = pgwas['estimates']#.loc[:,:,[f'y{i}.phenotype.proband' for i in range(5)]]
    beta_true = res['architecture'].beta_raw_diploid
    beta_hat = p_est[:,0,:]
    beta_true = np.hstack([np.zeros_like(beta_true),beta_true,np.zeros_like(beta_true)])
    sgwas = res['results_store'][GEN]['sgwas2']
    s_est = sgwas['estimates']#.loc[:,:,[f'y{i}.phenotype.proband' for i in range(5)]]
    sbeta_hat = s_est[:,0,:]
    OUTPUT.update(cov2dict(beta_hat.to_pandas().corr(), 'rbeta_hat_pgwas2', diag=False))
    OUTPUT.update(cov2dict(sbeta_hat.to_pandas().corr(), 'rbeta_hat_sgwas2', diag=False))
    ## GWAS 4
    pgwas = res['results_store'][GEN]['pgwas4']
    p_est = pgwas['estimates']#.loc[:,:,[f'y{i}.phenotype.proband' for i in range(5)]]
    beta_true = res['architecture'].beta_raw_diploid
    beta_hat = p_est[:,0,:]
    beta_true = np.hstack([np.zeros_like(beta_true),beta_true,np.zeros_like(beta_true)])
    sgwas = res['results_store'][GEN]['sgwas4']
    s_est = sgwas['estimates']#.loc[:,:,[f'y{i}.phenotype.proband' for i in range(5)]]
    sbeta_hat = s_est[:,0,:]
    OUTPUT.update(cov2dict(beta_hat.to_pandas().corr(), 'rbeta_hat_pgwas4', diag=False))
    OUTPUT.update(cov2dict(sbeta_hat.to_pandas().corr(), 'rbeta_hat_sgwas4', diag=False))
  
    return OUTPUT

oo = [parse_gen(gen) for gen in range(6)]
output = pd.concat([pd.DataFrame(oo[gen], index=[f"edu_seed{SEED}_{gen}"]) for gen in range(6)])

output.to_csv(f"{OPATH}/{PREFIX}_parsed.csv")
## save phenotypes

for gen in range(6):
    sim.results_store[GEN]['pop_GWAS']['estimates'].stack(cstat=['component','statistic']).to_pandas().to_csv(f"{OPATH}/{PREFIX}_pgwas{gen}.csv")
    sim.results_store[GEN]['sib_GWAS']['estimates'].stack(cstat=['component','statistic']).to_pandas().to_csv(f"{OPATH}/{PREFIX}_sgwas{gen}.csv")

pd.DataFrame.from_dict({'beta':sim.architecture.components[0].effects.beta_unscaled_unstandardized_diploid.ravel()}).to_csv(f"{OPATH}/{PREFIX}_effects.csv")



# output.loc[:,[x for x in output.columns if re.search('^r2_', x)]]
#for SEED in 1
# do     echo $SEED;     python3 edu_experiment_no_CD.py --path '/data1/ribo7412/edu_final' --seed $SEED; done
