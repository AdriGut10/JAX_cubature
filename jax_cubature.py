"""
jax_cubature.py

This module provides tools for numerical integration using JAX, including
initialization, basic rules, result ordering, and cubature calculations.

Functions:
    - initialise: Prepares and initializes the parameters for integration.
    - basic_rule: Applies the basic numerical integration rule.
    - order_results: Orders the results of integration for each subregion.
    - cubature: Performs advanced numerical integration using JAX.
"""

import numpy as np
import jax.numpy as jnp
import jax
from jax import lax
from functools import partial

jax.config.update("jax_enable_x64", True)

def initialise(ndim):
    twondim = 2.0**ndim

    lambda5 = 9.0/19.0   
    if ndim<=15: 
    #if ndim <= 2:
        rulcls = np.int64(2**ndim + 2*ndim*ndim + 2*ndim +1)
        lambda4 = 9.0/10.0
        lambda2 = 9.0/70.0
        weight5 = 1.0/(3.0*lambda5)**3 /twondim
    else:
        rulcls = np.int64(1 + (ndim*(12+(ndim-1)*(6+(ndim-2)*4)))//3)
        ratio = (ndim-2)/9.0
        lambda4 = (1.0/5.0 -ratio)/(1.0/3.0 -ratio/lambda5)
        ratio = (1.0 -lambda4/lambda5)*(ndim-1)*ratio/6.0
        lambda2 = (1.0/7.0 -lambda4/5.0 -ratio)/(1.0/5.0 -lambda4/3.0 -ratio/lambda5)
        weight5 = 1.0/(6.0*lambda5)**3

    weight4 = (1.0/15.0 -lambda5/9.0)/(4.0*(lambda4-lambda5)*lambda4**2)
    weight3 = (1.0/7.0 -(lambda5+lambda2)/5.0 +lambda5*lambda2/3.0)/(2.0*lambda4*(lambda4-lambda5)*(lambda4-lambda2)) -2.0*(ndim-1)*weight4
    weight2 = (1.0/7.0 -(lambda5+lambda4)/5.0 +lambda5*lambda4/3.0)/(2.0*lambda2*(lambda2-lambda5)*(lambda2-lambda4)) 

    if ndim<=15:
        weight1 = 1.0 -2.0*ndim*(weight2+weight3+(ndim-1)*weight4)-twondim*weight5
    else:
        weight1 = 1.0 -ndim*(weight2+weight3+(ndim-1)*(weight4+2.0*(ndim-2)*weight5/3.0))

    weight4p = 1.0/(6.0*lambda4)**2
    weight3p = (1.0/5.0 -lambda2/3.0)/(2.0*lambda4*(lambda4-lambda2)) -2.0*(ndim-1)*weight4p
    weight2p = (1.0/5.0 -lambda4/3.0)/(2.0*lambda2*(lambda2-lambda4))
    weight1p = 1.0 -2.0*ndim*(weight2p+weight3p+(ndim-1)*weight4p)

    ratio = lambda2/lambda4

    lambda5 = np.sqrt(lambda5)
    lambda4 = np.sqrt(lambda4)
    lambda2 = np.sqrt(lambda2)

    lambdas  = np.array([lambda2, lambda4, lambda5])
    weights  = np.array([weight1, weight2, weight3, weight4, weight5])
    weightsp = np.array([weight1p, weight2p, weight3p, weight4p])

    return rulcls,twondim,ratio,lambdas,weights,weightsp


def prepare_new_call(params,ndim):
    params['divflg'] = 0
    params['subrgn'] = params['rgnstr']
    params['wrkstr'] = params['wrkstr'].at[params['lenwrk']].set(params['wrkstr'][params['lenwrk']] - params['wrkstr'][params['subrgn']])
    params['finest'] = params['finest'] - params['wrkstr'][params['subrgn']-1]
    params['divaxo'] = jnp.int64(params['wrkstr'][params['subrgn']-2])

    for j in range(ndim): #Maybe optimize this with jax.lax.scan
        params['subtmp'] = params['subrgn']-2*(j+2)
        params['center'] = params['center'].at[j].set(params['wrkstr'][params['subtmp']+1])
        params['width']  = params['width'].at[j].set(params['wrkstr'][params['subtmp']])
        
    params['width']  = params['width'].at[params['divaxo']].set(params['width'][params['divaxo']]/2.0)
    params['center'] = params['center'].at[params['divaxo']].set(params['center'][params['divaxo']]-params['width'][params['divaxo']])
    return params


def basic_rule(functn,params,ndim):
    params['rgnvol'] = params['twondim']
    
    for j in range(ndim): #Optimize this with jax.lax.scan
        params['rgnvol'] = params['rgnvol']*params['width'][j]
        params['z'] = params['z'].at[j].set(params['center'][j])

    params['sum1'] = functn(params['z'])
    #Compute the symetric sums of functn(lambda2,0,0,..0) and functn(lambda4,0,0,..0), and 
    #maximum fourth difference
    params['difmax'] = -1.0
    params['sum2'] = 0.0
    params['sum3'] = 0.0
    for j in range(ndim): #Check if this can be optimized with jax.lax.scan
        params['z'] = params['z'].at[j].set(params['center'][j]-params['lambdas'][0]*params['width'][j])
        f1 = functn(params['z'])
        params['z'] = params['z'].at[j].set(params['center'][j]+params['lambdas'][0]*params['width'][j])
        f2 = functn(params['z'])
        params['widthl'] = params['widthl'].at[j].set(params['lambdas'][1]*params['width'][j])
        params['z'] = params['z'].at[j].set(params['center'][j]-params['widthl'][j])
        f3 = functn(params['z'])
        params['z']= params['z'].at[j].set(params['center'][j]+params['widthl'][j])
        f4 = functn(params['z'])
        params['sum2'] = params['sum2'] + f1 + f2
        params['sum3'] = params['sum3'] + f3 + f4
        df1 = f1+f2-2.0*params['sum1']
        df2 = f3+f4-2.0*params['sum1']
        params['dif'] = jnp.fabs(df1-params['ratio']*df2)
        
        def _if_update(params,j):
            def _update(_):
                return jnp.int64(j)
                
            def _no_update(_):
                return jnp.int64(params['divaxn'])
                
            divaxn = jax.lax.cond(params['difmax']<params['dif'],_update,_no_update,None)
            return divaxn
        params['divaxn'] = _if_update(params,j)

        def _if_update(difmax,dif):
            def _update(_):
                return dif
            def _no_update(_):
                return difmax
            difmax = jax.lax.cond(difmax<dif,_update,_no_update,None)
            return difmax
        
        params['difmax'] = _if_update(params['difmax'],params['dif'])
        params['z'] = params['z'].at[j].set(params['center'][j])

    def _if_cond(params):
        def _update(_):
            return jnp.int64((params['divaxo']+1)%ndim)
            
        def _no_update(_):
            return jnp.int64(params['divaxn'])
                    
        divaxn = jax.lax.cond(params['sum1'] == params['sum1']+params['difmax']/8.0,_update,_no_update,None)
        return divaxn
    params['divaxn'] = _if_cond(params)
    
    #Compute the symetric sums of functn(lambda4,lambda4,0,..0)
    params['sum4'] = 0.0
    for j in range(1,ndim):
        for k in range(j,ndim):
            for l in range(2):
                params['widthl'] = params['widthl'].at[j-1].set(-params['widthl'][j-1])
                params['z'] = params['z'].at[j-1].set(params['center'][j-1]+params['widthl'][j-1])

                for m in range(2):
                    params['widthl'] = params['widthl'].at[k].set(-params['widthl'][k])
                    params['z'] = params['z'].at[k].set(params['center'][k]+params['widthl'][k])
                    f1 = functn(params['z'])
                    params['sum4'] = params['sum4'] + f1
            
            params['z'] = params['z'].at[k].set(params['center'][k])
        params['z'] = params['z'].at[j-1].set(params['center'][j-1])

    #Compute symmetric sum of functn(lambda5,lambda5,lambda5,0,0...0)
    params['sum5'] = 0.0
    
    if ndim<=15:
    #if ndim<=2:
    #if False:
        params['widthl'] = -params['lambdas'][2]*params['width']
        params['z'] = params['center']+params['widthl']
        
        params['shrink'] = True
        def _loop(k,params):
            
            def _outer_if_true(params):
                params['shrink'] = False
                f1 = functn(params['z'])
                params['sum5'] = params['sum5'] + f1

                def _body_loop(j,params):
                    params['j'] = j
                    def _if_false(params):

                        def _inner_if_true(params):
                            params['widthl'] = params['widthl'].at[params['j']].set(-params['widthl'][params['j']])
                            params['z'] = params['z'].at[params['j']].set(params['center'][params['j']]+params['widthl'][params['j']])
                            return params

                        def _inner_if_false(params):
                            return params
                        
                        params = jax.lax.cond(params['flag'],_inner_if_true,_inner_if_false,params)
                        return params    

                    def _if_true(params):
                        
                        def _inner_if_true(params):
                            
                            params['widthl'] = params['widthl'].at[params['j']].set(-params['widthl'][params['j']])
                            params['z'] = params['z'].at[params['j']].set(params['center'][params['j']]+params['widthl'][params['j']])
                            params['flag'] = False
                            params['shrink'] = True
                            return params

                        def _inner_if_false(params):
                            return params
                        
                        params = jax.lax.cond(params['flag'],_inner_if_true,_inner_if_false,params)
                        return params
                    
                    params = jax.lax.cond(params['widthl'][j]<0.0,_if_true,_if_false,params)
                    return params


                params['flag'] = True
                params = jax.lax.fori_loop(0,ndim,_body_loop,params)
                return params
            
            def _outer_if_false(params):
                return params
            
            params = jax.lax.cond(params['shrink'],_outer_if_true,_outer_if_false,params)
            return params


        params = jax.lax.fori_loop(0,params['maxloop'],_loop,params)
    else:
        for j in range(ndim):
            params['widthl'] = params['widthl'].at[j].set(params['lambdas'][2]*params['width'][j])
        for i in range(2,ndim):
            for j in range(i,ndim):
                for k in range(j,ndim):
                    for l in range(2):
                        params['widthl'] = params['widthl'].at[i-2].set(-params['widthl'][i-2])
                        params['z'] = params['z'].at[i-2].set(params['center'][i-2]+params['widthl'][i-2])
                        for m in range(2):
                            params['widthl'] = params['widthl'].at[j-1].set(-params['widthl'][j-1])
                            params['z'] = params['z'].at[j-1].set(params['center'][j-1]+params['widthl'][j-1])
                            for n in range(2):
                                params['widthl'] = params['widthl'].at[k].set(-params['widthl'][k])
                                params['z'] = params['z'].at[k].set(params['center'][k]+params['widthl'][k])
                                f1 = functn(params['z'])
                                params['sum5'] = params['sum5'] + f1
                    
                        params['z'] = params['z'].at[k].set(params['center'][k])
                    params['z'] = params['z'].at[j-1].set(params['center'][j-1])
                params['z'] = params['z'].at[i-2].set(params['center'][i-2])

    #Compute fifth and seventh degree rules and error.
    params['rgncmp'] = params['rgnvol'] *(params['weightsp'][0]*params['sum1'] + params['weightsp'][1]*params['sum2'] + params['weightsp'][2]*params['sum3'] + params['weightsp'][3]*params['sum4'])
    params['rgnval'] = params['rgnvol'] *(params['weights'][0]*params['sum1'] + params['weights'][1]*params['sum2'] + params['weights'][2]*params['sum3'] + params['weights'][3]*params['sum4'] + params['weights'][4]*params['sum5'])
    params['rgnerr'] = jnp.abs(params['rgnval']-params['rgncmp'])

    params['finest'] = params['finest']+params['rgnval']
    params['wrkstr'] = params['wrkstr'].at[params['lenwrk']].set(params['wrkstr'][params['lenwrk']]+params['rgnerr'])
    params['funcls'] = params['funcls']+params['rulcls']
    return params


def order_results(params,ndim):
    
    def _place_first(params):
        #When divflg=0, start at top of list and move down
        #list tree to find correct position for results from 
        #first half of recently divided subregion
        params['subtmp'] = 2*params['subrgn'] +1
        
        def _body_while(_,params):
        
            def _outer_while_true(params):
                
                def _while_true(params):
                    
                    def _true(params):
                        params['sbtmpp'] = params['subtmp']+params['rgnstr']+1          
                        def _true_statement(_):
                            return params['sbtmpp']
                        def _false_statement(_):
                            return params['subtmp']
                        params['subtmp'] = jax.lax.cond(params['wrkstr'][params['subtmp']]<params['wrkstr'][params['sbtmpp']],_true_statement,_false_statement,params)               
                        return params['subtmp']
                    
                    def _false(params):
                        return params['subtmp']
        
                    params['subtmp'] = jax.lax.cond(params['subtmp']!=params['sbrgns']-1,_true,_false,params)

                    def loop_body(k,params):
                        params['wrkstr'] = params['wrkstr'].at[params['subrgn']-k].set(params['wrkstr'][params['subtmp']-k])
                        return params
                    
                    params = jax.lax.fori_loop(0,params['rgnstr']+1,loop_body,params)        

                    params['subrgn'] = params['subtmp']
                    params['subtmp'] = 2*params['subrgn'] +1
                    return params
                
                
                def _while_false(params):
                    return params

                cond2 = params['subtmp']<params['sbrgns'] 
                params = jax.lax.cond(cond2,_while_true,_while_false,params)
                return params
            

            def _outer_while_false(params):
                    return params


            cond1 = params['rgnerr']<params['wrkstr'][params['subtmp']]
            params = jax.lax.cond(cond1,_outer_while_true,_outer_while_false,params)
            return params

        params = jax.lax.fori_loop(0,params['maxorder'],_body_while,params)
        return params


    def _place_second(params):
    #When divflg=1, start at bottom right branch and move
    #up list tree to find correct position for results from
    #second half of recently divided subregion
        params['subtmp'] = ((params['subrgn']+1)//(2*(params['rgnstr']+1)))*(params['rgnstr']+1)-1
        
        def _body_while(_,params):
         
            def _outer_while_true(params):
                
                def _while_true(params):
                    
                    def _loop_body(k,params):
                        params['wrkstr'] = params['wrkstr'].at[params['subrgn']-k].set(params['wrkstr'][params['subtmp']-k])
                        return params
                    params = jax.lax.fori_loop(0,params['rgnstr']+1,_loop_body,params)

                    
                    params['subrgn'] = params['subtmp']
                    params['subtmp'] = ((params['subrgn']+1)//(2*(params['rgnstr']+1)))*(params['rgnstr']+1)-1
                    return params
                def _while_false(params):
                    return params
                cond1 = params['subtmp']>=params['rgnstr']
                params = jax.lax.cond(cond1,_while_true,_while_false,params)

                return params
            def _outer_while_false(_):
                return params
            
            cond2 = params['rgnerr'] > params['wrkstr'][params['subtmp']]
            params = jax.lax.cond(cond2,_outer_while_true,_outer_while_false,params)
            return params


        params = jax.lax.fori_loop(0,params['maxorder'],_body_while,params)
        return params
 

    cond = params['divflg']!=1
    params = jax.lax.cond(cond,_place_first,_place_second,params)

    #Store results of basic rule in correct position in list
    params['wrkstr'] = params['wrkstr'].at[params['subrgn']].set(params['rgnerr'])
    params['wrkstr'] = params['wrkstr'].at[params['subrgn']-1].set(params['rgnval'])
    params['wrkstr'] = params['wrkstr'].at[params['subrgn']-2].set(params['divaxn'])
    for j in range(ndim):
        params['subtmp'] = params['subrgn']-2*(j+2)
        params['wrkstr'] = params['wrkstr'].at[params['subtmp']+1].set(params['center'][j])
        params['wrkstr'] = params['wrkstr'].at[params['subtmp']].set(params['width'][j])
    return params


@partial(jax.jit, static_argnames=("functn","ndim","tol","maxpts","maxorder_pf","maxrule_pf"))#,"a","b",))
def jax_cubature(*, functn : callable, a : jnp.ndarray, b : jnp.ndarray, ndim : int ,tol : float = 1e-8, maxpts : int = 10000,  maxorder_pf : int = 1, maxrule_pf : int = 1) -> tuple:
    
    params = {}
    params['ndim'] = ndim
    params['a'] = a
    params['b'] = b
    params['maxpts'] = maxpts
    params['tol'] = tol 
    
    if ndim < 2:
        raise ValueError("ndim must be greater than 2")
    
    params['rgnstr']  = 2*ndim + 2
    params['divaxo']  = 0
    params['divaxn']  = 0

    #Compute the prefactors required by the cubature rule.
    rulcls,twondim,ratio,lambdas,weights,weightsp = initialise(ndim)
    
    params['rulcls'] = rulcls
    params['twondim'] = twondim
    params['ratio'] = ratio
    params['lambdas'] = lambdas
    params['weights'] = weights
    params['weightsp'] = weightsp


    params['lenwrk'] = (2*ndim+3)*(1+params['maxpts']//params['rulcls'])//2
    params['wrkstr'] = jnp.zeros(params['lenwrk']+1)
    params['funcls'] = 0
    
    
    #params['width']  = (params['b']-params['a'])/2.0
    #params['center'] = params['a'] + params['width']
    #params['num_neg'] = 0
    params['width']   = (params['b']-params['a'])/2.0
    params['num_neg'] = jnp.sum(params['width'] < 0)
    params['width']   = jnp.fabs(params['width'])
    params['center']  = (params['b']+params['a'])/2.0
    
    params['z'    ]   = jnp.zeros(ndim)
    params['widthl']  = jnp.zeros(ndim)
    params['rgnvol']  = 0.0
    params['sum1']    = 0.0
    params['sum2']    = 0.0
    params['sum3']    = 0.0
    params['sum4']    = 0.0
    params['sum5']    = 0.0
    params['dif']     = 0.0
    params['difmax']  = 0.0
    params['shrink']  = True
    params['flag']    = True
    params['j']       = 0

    params['rgncmp'] = 0.0
    params['rgnval'] = 0.0

    params['finest'] = 0.0
    params['rgnerr'] = 0.0

    params['subrgn'] = params['rgnstr']
    params['sbrgns'] = params['rgnstr']+1
    params['subtmp'] = 0
    params['sbtmpp'] = 0
    params['divflg'] = 1  
    params['relerr'] = 1.0

    params['maxloop']  = params['rulcls']*maxrule_pf
    params['maxorder'] = params['lenwrk']*maxorder_pf


    #Initial call to basic rule
    params = basic_rule(functn,params,ndim)
    #Order and store results of basic rule
    params = order_results(params,ndim)
    #Check the convergence for possible termination.
    params['relerr'] = jnp.where(jnp.fabs(params['finest']) != 0.0, params['wrkstr'][params['lenwrk']] / jnp.fabs(params['finest']), 1.0)
    
    def loop_cond(params):
        return params['relerr'] > params['tol']

    def update_state(params):
        #Prepare for new call to basic rule
        params  = prepare_new_call(params,ndim)
        #Call basic rule in the first subregion
        params = basic_rule(functn, params,ndim)
        #Order and store results of basic rule
        params = order_results(params,ndim)

        #Prepare a new call to basic rule in the second subregion
        params['center'] = params['center'].at[params['divaxo']].set(params['center'][params['divaxo']] + 2.0 * params['width'][params['divaxo']])
        params['sbrgns'] = params['sbrgns'] + params['rgnstr'] + 1
        params['subrgn'] = params['sbrgns'] - 1

        #Call basic rule in the second subregion
        params = basic_rule(functn, params,ndim)
        #Order and store results of basic rule
        params = order_results(params,ndim)
        #Check the convergence for possible termination.
        params['relerr'] = jnp.where(jnp.abs(params['finest']) != 0.0, params['wrkstr'][params['lenwrk']] / jnp.abs(params['finest']), 1.0)
        return params

    
  
    def loop_step(params, _):
        # The second argument is unused; it's just there because lax.scan requires it
        return jax.lax.cond(loop_cond(params), update_state, lambda x: x, params ), None

    
    params, _ = jax.lax.scan(loop_step, params, xs=None, length=maxpts//(2*rulcls -2))
    params['finest'] = params['finest']*(-1)**params['num_neg']
    return params['finest'], params['relerr'] 


if __name__ == "__main__":
    def fun(x_array):
        x = x_array[0]
        y = x_array[1]
        z = x_array[2]
        return x**2 +jnp.log10(y+2)**2.5 + x*z**jnp.log(2)
    a_jax = jnp.array([0, 0, 0])
    b_jax = jnp.array([jnp.pi, jnp.pi, 1])  
    result, error = jax_cubature(functn=fun, a=a_jax, b=b_jax, ndim=3)
    print("Result : ",result)
    print("Estimated error : ",error)

    def f(x):
        a_jax = jnp.array([0, 0, 0])
        b_jax = jnp.array([jnp.pi, jnp.pi, x])  
        result, error = jax_cubature(functn=fun, a=a_jax, b=b_jax, ndim=3)
        return result
    
    df = jax.jacfwd(f)
    print("Gradient : ",df(1.0))