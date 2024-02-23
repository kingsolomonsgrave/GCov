#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:08:58 2019

@author: Mauri GCov Functions
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 21:05:36 2019

@author: M Hall
"""




def GCov_p(theta, 
            data, 
            lags = None,
            H = None,
            mode = 'standard',
            while_print = False, 
            check= False,
            manual = False,
            debug = False,
            include_time = False):
    
    """
    This function calculates the value of the objective function up to p lags (p is a positive inteter).
    
    
    Parameters
    
    format
    
    name: type (default value)
    
    The term in brackets above is not included if the default is None
    
    ---------------------------
    theta: np.array
        The parameter values to be optimized in the GCov
        
    data: pd.DataFrame object
        A df with TWO columns!
        
    lags: int
        Number of lags to include in the VAR
        
    H: int
        Number of coerrlation lags


    
    mode: str
        Sets the nonlinear function to be used for calculating coefficients
         'standard': each lag is raised to consecutive powers
                     For example if lag = 1 then the function is square
                     if lag = 2 then the function is square and then cubed
        'exp' : each lag is exponentiated (e^x)
        
        '3rd power, 4th power, 5th power' : these are like standard except that 
        powers increase much faster, so the first lag would be cubed, then to the 6th for 
        '3rd power' etc
        
        'factorial' powers of lags increase as a factorial starting at lag = 
        with power 6, then 24 at lag =2 etc
    

    
    while_print: bool (False)
        If True then the function will print during the while loop
        
  
    
    
    """
    

    
    
    
    #### Main Function ###
   
    import numpy as np
    import pandas as pd
    
    
        
    if isinstance(lags, int) and (lags >= 0):
        pass
    else:
        raise Exception('lags must be a postive integer value')
        
    if isinstance(data, pd.DataFrame):
        data = data.values #make into a numpy array
    
    n = len(data)
    
    n_cols = data.shape[1]
    n_cols_sq = n_cols**2
    
    
    eps_cols = n_cols*(lags + 1)
    eps = np.zeros((n-lags, eps_cols))
    
    
    phi = np.empty(n_cols_sq*lags) # dtype = object
    for i in range(n_cols_sq*lags):
        phi[i] = theta[i]
    
    
    PHI = [np.nan for i in range(lags)]
    
    
    for i in range(lags):
        PHI[i] = phi[i*n_cols_sq : n_cols_sq*(i + 1)].reshape(n_cols,n_cols)
    
    
    y_lagged = [np.nan for i in range(lags +1)]
    if lags != None:
        for L in range(lags+1):
            y_lagged[L] = data[lags - L:len(data) - L, :]
    y_lagged = np.concatenate(y_lagged, axis = 1)
    
  

    sum_of_lagged_phi = 0
    for i in range(lags):
        lagged_phi = PHI[i].dot(y_lagged[:,n_cols*(i + 1): n_cols*(i + 2)].T)
        
        #lagged_phi = PHI[i].dot(y_lagged[i+1].T)
        sum_of_lagged_phi =  sum_of_lagged_phi + lagged_phi
    #ps = y_lagged[0].T - sum_of_lagged_phi
    ps = y_lagged[:, 0:data.shape[1] ].T - sum_of_lagged_phi
    
    
    eps[:, 0:n_cols] = ps.T
    
    

    

    if mode == 'standard':
        for i in range(1, lags + 1):
           eps[:, i*n_cols :(i + 1)*n_cols] =  eps[:, 0:n_cols]**(2*i)
           
    elif mode == 'exp':
        for i in range(1, lags + 1):
            eps[:, i*n_cols :(i + 1)*n_cols] =  np.exp(eps[:, 0:n_cols]**i)
            
            
    elif mode == '3rd power':
        for i in range(1, lags + 1):
          
            eps[:, i*n_cols :(i + 1)*n_cols] =  eps[:, 0:n_cols]**(3*i)
          
    elif mode == '4th power':
        for i in range(1, lags + 1):
       
            eps[:, i*n_cols :(i + 1)*n_cols] =  eps[:, 0:n_cols]**(4*i)
           
    elif mode == '20th power':
        for i in range(1, lags + 1):
            eps[:, i*n_cols :(i + 1)*n_cols] =  eps[:, 0:n_cols]**(20*i)
    
    elif mode == 'factorial':
         for i in range(1, lags + 1):
             
             if np.math.factorial(i+2) < 120:
                 eps[:, i*n_cols :(i + 1)*n_cols] =  eps[:, 0:n_cols]**(np.math.factorial(i+2))
             
             else:
                 eps[:, i*n_cols :(i + 1)*n_cols] =  eps[:, 0:n_cols]**(np.math.factorial(3)*2*i)
               
    else:
        raise Exception('mode not recognized')
            
    
    
    for i in range(eps.shape[1]):
        eps[:, i] = eps[:, i] - np.mean(eps[:, i])
        
    
    corr = np.zeros((eps_cols, eps_cols))
    vc = np.cov(eps.T)
    did = np.diag(vc)
    di = np.sqrt(did)
    di = di.reshape((eps_cols,1))
    vov = di.dot(di.T)
    
  
    if check == False:
    
        count1 = 1
        while count1 < H:
            eps1 = eps[0: len(eps) - count1, :]
            
            n1 = len(eps1)
            epl2 = eps[count1:, :]
            co = (eps1.T.dot(epl2))/n1
            co = np.divide(co, vov)
            sco = co * co
            corr = corr + sco
            
            count1 = count1 + 1
            if while_print:
                print('when h is ' + str(count1) + ' sum corr is ', np.sum(corr))
        s = np.sum(corr)
        #print(s)
        return s
    else:
        pass
    

        


#### Gives output of coefficients and residuals, acfs and eigenvalues


def GCov_General_phi(fun = GCov_p, 
                     x0 = None  , 
                     args = None, 
                     H = None,
                     m = 'BFGS',
                     zero = True, 
                     OLS = True, 
                     VAR_lags = None,
                     imports = True,
                     print_start = False,
                     eigen = False,
                     resids = False,
                     coefs = True,
                     out = False,
                     path = None,
                     acf = False,
                     sq= False,
                     M = 6,
                     N = 16, 
                     durbin_watson = True,
                     warnings_off = True,
                     resids_return = False,
                     acf_lags = 20,
                     verbose = False, 
                     block_matrix = False,
                     NC_resid = False,
                     mode = 'standard', 
                     std_err = False):
    """Calculates the Phi 2 by 2 matrix using the GCov.
    
    Parameters:
        
    NOTE: default = None if not otherwise assigned
    
    NOTE: default appears in brackets after the type
    
    fun: function (GCov_p4)
        the function to be minimized (bivariate time series)
    
    x0: list or tuple 
        starting values, calculated automatically unless specified otherwise
    
    args: tuple
        a tuple containing arguments (data, , lags, H), must be given by user
        
    H: int
        Number of autocorrelation lags used in estimation
        
    m: str (default 'BFGS')
        optimation method, eg 'BFGS'
    
    zero: bool (False)
        If True then the function uses the zero vector as the starting values
        
    OLS: bool (True)
        If True then the function uses OLS estimation for starting values
                     
    VAR_lags: int
        Number of lags to use in the estimation (1 to 4)
    
    imports: bool (True)
        If True needed packages are imported
        
    print_start: bool (False)
        If True then the start values are printed
        
    eigen: bool (False)
        If True then the function returns the eigen values for the estimated matricies
        NOTE: resids must be False to get this output!
        
    resids: bool (False)
        If True the function outputs a pd.DataFrame of residuals
    
    coefs: bool  (True)
        If True then coefficients are output (np.arrays)
    
    out: bool (False)
        If True then a jpg is output to path
    
    path: str
        The file path for the jpg
                    
    acf: bool (False):
        If True then an ACF will be output
    
    sq: bool (False):
        When resids is True & ACF is True then
        When this is True the ACF of the SQUARED residuals is returned
    
    M: int (6)
        One of two sizes of the figure created, the first
                     
        
    N: int (16)
        One of two sizes of the figure created, the second
        
    durbin_watson: bool:
        Returns Durbin-Watson statistic for the residuals
        Get this by setting resids to True and ACF to False
        
    warnings_off: bool (True
        If True then warnings are turned off
    
    resids_return: bool (False)
        If True then the function will output the residuals as a pd.DataFrame object
        
    acf_lags: int 
        number of lags to use in the ACF graph
        
    block_matrix: bool
        if True the function returns a block matrix of coefficients with matricies along the first row,
        Identity matricies along the diagional and zeros everywhere else
    """
    # Imports
    lags = VAR_lags
    if warnings_off:
        import warnings
        warnings.filterwarnings("ignore")
    if imports:
        import GCov
        from statsmodels.tsa.api import VAR
        import numpy as np
        import pandas as pd
        import scipy as scipy
        from statsmodels.stats.stattools import durbin_watson as dw
        import seaborn as sns
        from statsmodels.graphics.tsaplots import plot_acf
        import matplotlib.pyplot as plt
    
    ### Inner Functions ###
    def eig(x, eigv = True):
        y = np.linalg.eig(x)
        if eigv:
            return y[0]
        else:
            return y
    
    data = args[0]
    n_cols = data.shape[1]
    n_cols_sq = n_cols**2
    
    
    if isinstance(data, pd.DataFrame):
        col_names = [np.nan for i in range(len(data.columns))]
       
            
        for i in range(len(data.columns)):
            col_names[i] = data.columns[i]
        
        #for ind in range(len(col_names)):
            #list_of_variables[ind] = col_names[ind]
        

        
       #OLD .values, new: to_numpy() 
    #try:
        #data = args[0].values # as ndarray
    try:
        data = args[0].to_numpy()# as ndarray
    except:
        pass

    if isinstance(data, np.ndarray):
        n_cols = data.shape[1]
    elif isinstance(data, pd.DataFrame):
        n_cols = len(data.columns)
    
  
    if VAR_lags == None:
        VAR_lags = 1
      
        lags = VAR_lags
        
    if zero == False: #If zero is false and OLS is false then the user must supply starting values 
        if OLS: # if zero is false but OLS is True then the function calculates and uses OLS starting values
            if VAR_lags != None:
                A = VAR(data).fit(VAR_lags, trend = 'nc').coefs
                
                start = A.flatten()
                if print_start:
                    print("The OLS starting values are ", start)
                
            
        else:
            start = x0
            
    
    else:
        start = []
        for i in range(VAR_lags*n_cols_sq):
            start = start + [0]
        if print_start:
            print('start is ', start)
    
    phi = scipy.optimize.minimize(fun, x0 = start, method = m, args = args).x
    
    phi_flat = phi.flatten()
    
    if coefs == False:
        if std_err:
            se_pre =  np.diag(scipy.optimize.minimize(fun, x0 = start, method = m, args = args).hess_inv)
            se = np.sqrt(se_pre/len(data))
            return se
    PHI = [i for i in range(VAR_lags)]
    for i in range(VAR_lags):
        PHI[i] = phi_flat[i*n_cols_sq : i*n_cols_sq + n_cols_sq].reshape(n_cols,n_cols)
   
    
    
    #y_lagged = np.empty(lags + 1, dtype = object)
    y_lagged = [np.nan for i in range(lags +1)]
    if resids:
        coefs = False
        for L in range(lags+1):
            y_lagged[L] = data[lags - L:len(data) - L, :]
            #y_lagged[i] = data[i:len(data) - lags + i]
            
        y_lagged = np.concatenate(y_lagged, axis = 1)
        
        sum_of_lagged_phi = 0
        
        for i in range(lags):
            lagged_phi = PHI[i].dot(y_lagged[:,n_cols*(i + 1): n_cols*(i + 2)].T)
            #lagged_phi = PHI[i].dot(y_lagged[i+1].T)
            sum_of_lagged_phi +=  lagged_phi
        
        resi = y_lagged[:,0:n_cols].T - sum_of_lagged_phi
        resi = resi.T
        
        if acf:
            #import seaborn as sns
            #sns.set_style("darkgrid")
            coefs = False
            eigen = False
            with sns.axes_style("darkgrid"):
                #fig, ax = plt.subplots(figsize=(N, M))
                #ax.grid()
                #fig2, ax2 = plt.subplots(figsize=(N, M))
                #ax2.grid()
                if sq:
                    for i in range(len(col_names)):
                        fig, ax = plt.subplots(sharey= True)
                        plot_acf(resi[:,i]**2, lags = acf_lags, title = col_names[i] + " squared", ax = ax)
                else:
                    #plt.subplots()
                    for i in range(len(col_names)):
                        
                        fig, ax = plt.subplots(sharey= True)
                        plot_acf(resi[:,i], lags = acf_lags, title = col_names[i], ax = ax)
                    
            if out:
                if path == None:
                    print("You must supply a string filepath to save the file")

                else:
                    plt.savefig(path)
       
            plt.show()
                
        if resids_return:
            if sq:
                return resi*resi
            else:
                return resi
                
        if durbin_watson:
            if verbose:
                print('dw near 2 implies lack of serial correlation')
            list_of_dw = []
            for i in range(resi.shape[1]):
                if sq:
                    list_of_dw.append(dw(resi[:,i]**2))
                else:
                    list_of_dw.append(dw(resi[:,i]))
                
            return list_of_dw
        
    if eigen:
            
        L = [eig(PHI[i]) for i in range(lags)]
        return L
    
        
    elif coefs:
        if lags ==1:
            return PHI
        else:
            list_of_coefs = [PHI[i] for i in range(lags)]
            array_of_coefs = np.asarray(list_of_coefs)
        if block_matrix:
            pass
        else:
            return array_of_coefs



            
        
   


def mh_boot_residuals(data, 
                      n_samples = None, 
                      lags = 1, 
                      H = 11,
                      test = False, 
                      imports = True, 
                      bells = 0,
                      std_err = False, 
                      block = False, 
                      outlier_tol = None,
                      try_zero = True,
                      print_reps_mod = 25,
                      verbose = True,
                      start_zero = False, 
                      OLS = False,
                      print_iter = True,
                      OLS_boot = False,
                      coef_boot = True,
                      irf = False,
                      irf_lags = None, 
                      col = 0, 
                      cross = False,
                      alpha = 0.05,
                      first_on_second = True,
                      OLS_irf = False,
                      zero = True,
                      NC_boot = False,
                      mixed = True,
                      linked_resids = True,
                      diag = True):
    
    assert lags != None
    assert n_samples != None
    assert outlier_tol != None
    
    
    
    if imports:
        import numpy as np
        from GCov import GCov_General_phi
        from sklearn.utils import resample
        from playsound import playsound
        import pandas as pd
        from statsmodels.tsa.vector_ar.var_model import VAR as VAR
          
        
    if isinstance(data, np.ndarray):
        
        n_cols = data.shape[1]
        
    elif isinstance(data, pd.DataFrame):
        n_cols = len(data.columns)
        
    n_cols_sq = n_cols**2
    
    n_samples = n_samples + 1 #increment number of samples for computational reasons
    
    if verbose:
        print('data type of the input data is', type(data))
        print('calculating resids (once)')
        

    if OLS_boot:
        resids = VAR(data).fit(lags, trend = 'nc').resid
    
        
       
    else:
        
        resids = GCov_General_phi(fun = GCov_p, 
                              args = (data, lags, H, 'standard'), VAR_lags = lags, coefs = 0, resids = 1, resids_return = 1, zero = zero)

        
        
    resids = resids - np.mean(resids) #Demean the residuals
    
    if verbose:

        print('calculating phi (once)')
    
    if OLS_boot:
        phi  = VAR(data).fit(lags, trend = 'nc').coefs
    else:
        if verbose:
                print('using GCOV coeficients for the Boot')
        phi = GCov_General_phi(fun = GCov_p, 
                              args = (data, lags, H, 'standard'), VAR_lags = lags, coefs = 1, resids = 0, resids_return = 0, OLS =1, zero = zero)
   
        
    PHI = phi
    if verbose:
        
        print('PHI is ', PHI)
        
   
    

    y_lagged = [np.nan for i in range(lags +1)]
    
    if lags != None:
        data = np.asarray(data)
        for L in range(lags+1):
            y_lagged[L] = data[lags - L:len(data) - L, :]
            #y_lagged[i] = data[lags - i:len(data) - i, :]
    else:
        raise Exception('lags must be an integer')
        

    y_lagged = np.concatenate(y_lagged, axis = 1)
        
    coefs3 = np.zeros(lags*n_cols_sq)

    
    if verbose:
        print('staring boot lags = ' + str(lags))
    count = 1
    
    if coef_boot:
    
        while count < n_samples:
            #if print_iter:
               # print('count is', count)
            if count%print_reps_mod == 0:
                if print_iter:
                    if block:
                        print('on iteration ' + str(count) + ' using blocks on the residuals')
                    else:
                        print('on iteration ' + str(count) + ' NOT using blocks')
            
            re_resids = resample(resids)
        
            try:
                re_resids = re_resids.values
            except:
                pass
            
            sum_of_lagged_phi = 0
            for i in range(lags):
                #lagged_phi = PHI[i].dot(y_lagged[:, i+2:i+4].T)
                lagged_phi = PHI[i].dot(y_lagged[:,n_cols*(i + 1): n_cols*(i + 2)].T)
                sum_of_lagged_phi =  sum_of_lagged_phi + lagged_phi
            
           
            
            new_data = y_lagged[:,0:n_cols].T + sum_of_lagged_phi + re_resids.T
                
            
         
            new_data = new_data.T
        
            if OLS_boot:
                coe = VAR(new_data).fit(lags, trend = 'nc').coefs
            else:
                try:
                    coe = GCov_General_phi(fun = GCov_p, 
                                  args = (new_data, lags, H, 'standard'), VAR_lags = lags, coefs = 1, resids = 0, resids_return = 0, zero = zero)
                
                except:
                    raise Exception('coefs were not calculated')
                    
            
            condition1 = np.abs(np.linalg.eig(coe[0])[0][0]) > 1
            condition2 = np.abs(np.linalg.eig(coe[0])[0][1]) < 1
            condition3 = np.abs(np.linalg.eig(coe[0])[0][0]) < 1
            condition4 = np.abs(np.linalg.eig(coe[0])[0][1]) > 1
            
            if np.all(np.abs(coe) < outlier_tol):
                #print('coefs too big')
                if mixed:
                    if np.all(condition1 and condition2) or np.all(condition3 and condition4):
                        coe = np.asarray(coe)
                        row2 = coe.flatten()
                        coefs3 = np.vstack((coefs3, row2))
                        count = count + 1
                else:
                    coe = np.asarray(coe)
                    row2 = coe.flatten()
                    coefs3 = np.vstack((coefs3, row2))
                    count = count + 1
            
                
            else:
                pass
                #if zero == 1:
                #    new_zero = 0
                #elif zero == 0:
                #    new_zero = 1
                #coe2 = GCov_General_phi(fun = GCov_p, 
                #                      args = (new_data, lags, H), VAR_lags = lags, coefs = 1, resids = 0, resids_return = 0, zero = new_zero)
                        
               
                
                #if verbose:
                #    print('zero is ', zero, 'new zero is ' , new_zero)
                
                #condition1 = np.abs(np.linalg.eig(coe2[0])[0][0]) > 1
                #condition2 = np.abs(np.linalg.eig(coe2[0])[0][1]) < 1
                #condition3 = np.abs(np.linalg.eig(coe2[0])[0][0]) < 1
                #condition4 = np.abs(np.linalg.eig(coe2[0])[0][1]) > 1
                    
#                if np.all(np.abs(coe2) < outlier_tol):
#                    if mixed:
#                        if np.all(condition1 and condition2) or np.all(condition3 and condition4):
#                            if verbose:
#                                print('Zero vector worked where the OLS failed!')
#                            coe2 = np.asarray(coe2)
#                            row2 = coe2.flatten()
#                            coefs3 = np.vstack((coefs3, row2))
#                            count = count + 1
#                    else:
#                         coe2 = np.asarray(coe2)
#                         row2 = coe2.flatten()
#                         coefs3 = np.vstack((coefs3, row2))
#                         count = count + 1
                 
                    

         
        coefs3_df = pd.DataFrame(coefs3)
        list_of_phi = ['Phi' + str(i) for i in range(1, n_cols_sq*lags+1)]
        
        
        coefs3_df.columns = list_of_phi
        
    
        coefs3_df = coefs3_df.tail(len(coefs3_df) - 1)
            
        for i in range(bells):
            
            playsound('/Users/Frida/Desktop/A-Tone-His_Self-1266414414.wav')
                



        
        
        
    if diag:
        VAR_boot_diagnostics(coefs3_df.dropna())
    else:
        return coefs3_df.dropna()
            
    
            
                  
    
            
                
    

def std_err_VAR(boots):
 
    def std_err(boots, col = None):
        """
        input: boots (list)

        A list of dataframes with bootstraps samples
        """
        import numpy as np
        from  scipy.stats import jarque_bera as jb
        list_of_means = []
    
        for i in range(len(boots)):
            list_of_means.append(np.mean(boots[i][col]))
        num = np.mean(list_of_means)

        #denom = np.std(list_of_means)
        #print('t-stat:    ', num/denom)
        print('p value:    ', p_value(np.array(list_of_means)))
        print('jb p value:   ', jb(list_of_means)[1])
        print('mean of means:' ,num)
        
    for phi in ['Phi' + str(i) for i in range(1,len(boots) + 1)]:
        print(phi)
        std_err(boots, phi)
        print("        ")  
        

   
def SimNCVAR_Bivariate_Causal(a = 0.7,b = -0.3,c = 0.0,d = 0.5, 
                              e = 0.0, f = 0.0, g = 0.0, h = -0.0,
                    length = 150,
                    col_name1 = 'BTC',
                    col_name2 = 'ETH',
                    df = True, return_phi = False, cauchy = False, return_innovations = False):

### simulating a VAR process ####
    import numpy as np
    import pandas as pd
    
    PHI = np.array([[[a, b], [c, d]], [[e, f],[g, h]]])
    if return_phi:
        return PHI
        raise Exception('after returning PHI the problem is halted')
    innovations = np.random.randn(2, length)
    if cauchy: 
        innovations2 = np.empty_like(innovations)
        innovations2[0] = np.random.standard_cauchy(length)
        innovations2[1] = np.random.standard_cauchy(length)
        innovations = innovations2
    else:
        pass
        
    
    
    
    Y_t = np.empty_like(innovations)

    for t in range(len(innovations[0]*2)):
        
        if t == 0:
            Y_t[0][0] = 1
            Y_t[1][0] = 1    
        else:
            Y_t[0][t] = PHI[0][0][0]*Y_t[0][t-1] + PHI[0][0][1]*Y_t[1][t-1] + PHI[1][0][0]*Y_t[0][t-2] + PHI[1][0][1]*Y_t[1][t-2] + innovations[0][t]
            Y_t[1][t] = PHI[0][1][0]*Y_t[0][t-1] + PHI[0][1][1]*Y_t[1][t-1] + PHI[1][1][0]*Y_t[0][t-2] + PHI[1][1][1]*Y_t[1][t-2] + innovations[1][t]
            
    Yt_df = pd.DataFrame(Y_t.T).dropna()
    Yt_df.columns = [col_name1, col_name2]
    Yt_df.tail(length)
    
    if df:
        
        if return_innovations:
            return Yt_df, innovations.flatten()
        else:
            return Yt_df
    else:
        
        if return_innovations:
            return Yt_df.values, innovations.flatten()
        else:
            return Yt_df.values
    


def acf(data, title = None, lags = 20):
    from statsmodels.graphics.tsaplots import plot_acf as plot_acf
    import matplotlib.pyplot as plt
  
    if title == None:
        
        plot_acf(data, lags = lags)
       
    else:
       
        plot_acf(data, lags = lags, title = title)
        
    
    

def create_variable_names(var, n):
    """
    paramters
    
    ---------------
    args:
    
    var: str
        name of the variable name to create
        
    n: int
        number of variables to create
        
    Returns:
        list of variable names
    ----------------
        
    EXAMPLE:
    
    create_variables(var = 'Phi', n = 2) will return
    
    ['Phi1', 'Phi2']
    
    """
    
    
    if isinstance(var, str) and isinstance(n, int):
        n = n+1
        L = []
        for i in range(1,n):
            x = str(var) + '_' + str(i)
            L.append(x)
        return L 
    


def rmv_string(df, x):
    """Takes a dataframe column and attempts to turn strings into floats

    Parameters:
    df (pandas dataframe): 
        
    x (str) : name of column to be changed into float type from string type

    Returns:
    nothing, this function just modifies the dataframe

   """
    df[x] = df[x].str.replace(r',', r'').astype('float')     
    


### From http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016 ###
def tsplot(y, lags=30, figsize=(10, 8), style='bmh', cust_title = None):
    """Input a time series and this funciton outputs a set of graphs pertaining to the time series

    Parameters:
    y (dataframe): datafame of the time series
    
    figsize (tuple): dimension of the graph to be created
    
    style (str): string of the output style
    
    cust_title (str) : Default is None, if given the graph has this title
    
    Returns:
    a graph

   """
    import pandas as pd

    
    
    import statsmodels.tsa.api as smt
    import statsmodels.api as sm
    import scipy.stats as scs
    #from arch import arch_model

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        if cust_title != None:
            ts_ax.set_title('Time Series Analysis Plots ' + cust_title)
        else:
            ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 
    


        
def mh_acf(data, 
           maxlags = 30, 
           cols = None, 
           panel  = 'four',  
           alpha_graph = 0.4, 
           alpha_conf = 0.05,
           var_names = None,
           ymax = 1, 
           ymin = -1.0,
           save = None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import acf
    from statsmodels.tsa.stattools import ccf
    #n_cols = data.shape[1]
    
    if var_names == None:
        var_names = ['BTC', 'ETH']
    

    
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    
    
    maxlags = maxlags + 1
    #norm = max(np.correlate(data[:,cols[0]], data[:,cols[1]], mode ='full'))
    #half = int(np.correlate(data[:,cols[0]], data[:,cols[1]], mode ='full').__len__()/2)
    
    def confint(data_input, acf_values = None, n_lags = None, alpha_conf = alpha_conf, reverse = False):
        import numpy as np
        from scipy.stats import norm as sci_norm
        #from statsmodels.compat.python import (iteritems, range, lrange, string_types,lzip, zip, long)
        from statsmodels.compat.python import lzip
        #from statsmodels.tsa.stattools import acf
        #from statsmodels.tsa.stattools import ccf
        
        nobs = len(acf_values)
        varacf = np.ones(nobs) / nobs
        varacf[0] = 0
        varacf[1] = 1. / nobs
        varacf[2:] *= 1 + 2*np.cumsum(acf_values[1:-1]**2)
        interval = sci_norm.ppf(1 - alpha_conf/2)*np.sqrt(varacf) 
        #interval = 1.96**np.sqrt(varacf)
       # print(sci_norm.ppf(1 - alpha/2))
        #half_acf = 0 #int(len(acf_values)/2)
        #acf = 0 #acf_values/acf_values[half_acf: half_acf + 1]

    
        confinterval = np.array(lzip(-interval, interval))
        if reverse:
            #return confinterval[:maxlags,0][::-1], confinterval[:maxlags,1][::-1]
            return confinterval[:, 0][::-1], confinterval[:,1][::-1]
        else:
            #return confinterval[:maxlags,0], confinterval[:maxlags,1]
            return confinterval[:,0], confinterval[:,1]
    
    if panel == 'four':
        cols = (0,1)
        x = data[:,cols[0]]
        y = data[:,cols[1]]
        #norm = np.sqrt(np.dot(x, x) * np.dot(y, y))
        
        name1 = var_names[0]
        name2 = var_names[1]
        
        plt.figure(figsize = (12,8), dpi = 150)
        
        plt.subplot(221)
        
        
        #half = int(np.correlate(x,x, mode ='full').__len__()/2)
     
        
        #acf_values_all = np.correlate(x,x, mode ='full')
        
        #acf_values_maxlags = acf_values_all[half:half + maxlags]
        acf_values_CI = acf(x, nlags = len(data))
        acf_values_stats = acf(x, nlags = maxlags)
        #norm = max(np.abs(acf_values_all))
        #norm = np.sqrt(np.dot(x, x) * np.dot(y, y))
        #acf_values_maxlags_norm = acf_values_maxlags/norm
        
        #acf_values_all_norm = acf_values_all/norm
        cL, cH = confint(data_input = data, acf_values = acf_values_CI)
        cL = cL[:maxlags+2]
        cH = cH[:maxlags+2]
        x_index = np.arange(1, len(cL) ).flatten()
        plt.fill_between(x_index,cL[1:], cH[1:], color='black', alpha= alpha_graph)
        
        z = acf_values_stats

        plt.stem(z, linefmt = 'black', markerfmt='ko', use_line_collection = True)
       
        
        #axes.set_ylim([-0.2,1])
        plt.title('ACF of ' + name1)
        
        
        plt.subplot(222)
        
        acf_values_CI = ccf(x, y)
        acf_values_stats = ccf(x,y)[:maxlags]
     
        
        cL, cH = confint(data_input = data, acf_values = acf_values_CI)
        cL = cL[:maxlags+2]
        cH = cH[:maxlags+2]
        x_index = np.arange(1, len(cL) ).flatten()
        plt.fill_between(x_index,cL[1:maxlags+2], cH[1:maxlags+2], color='grey', alpha = alpha_graph)
        
        z = acf_values_stats

        plt.stem(z, linefmt = 'black', markerfmt='ko', use_line_collection = True)
        axes = plt.gca()
        
        axes.set_ylim([ymin, ymax])
        plt.title('Cross ACF of ' + name1 + ' & ' + name2)
    
        
        plt.subplot(223)
        
        acf_values_CI = ccf(y, x)
        acf_values_stats = ccf(y,x)[:maxlags]
        acf_values_stats = np.array(acf_values_stats)
        acf_values_stats = acf_values_stats[::-1]
     
        
        cL, cH = confint(data_input = data, acf_values = acf_values_CI, reverse = 0)
        cL = cL[:maxlags+2]
        cH = cH[:maxlags+2]
        
        
       
        x_index = np.arange(-len(cL), -2).flatten()
       
        plt.fill_between(x_index,cL[1:maxlags+1][::-1], cH[1:maxlags+1][::-1], color='grey', alpha = alpha_graph)
        
        z = acf_values_stats
      
        plt.stem(x_index ,z, linefmt = 'black', markerfmt='ko', use_line_collection = True)
        axes = plt.gca()
        
        axes.set_ylim([ymin, ymax])
        plt.title('Cross ACF of ' + name2+ ' & ' + name1)
       
        

        
        plt.subplot(224)
      
        acf_values_CI = acf(y, nlags = len(data))
        acf_values_stats = acf(y, nlags = maxlags)
      
        cL, cH = confint(data_input = data, acf_values = acf_values_CI)
        cL = cL[:maxlags+2]
        cH = cH[:maxlags+2]
        x_index = np.arange(1, len(cL) ).flatten()
        plt.fill_between(x_index,cL[1:], cH[1:], color='black', alpha= alpha_graph)
        
        z = acf_values_stats

        plt.stem(z, linefmt = 'black', markerfmt='ko', use_line_collection = True)
       
      
        plt.title('ACF of ' + name2)
        if save != None:
            try:
                plt.savefig(save, dpi = 150)
            except:
                print('Image not saved to ', save)
        

    
    
        
#        elif panel == 'multi':
#            plt.figure(figsize = (12,10), dpi = 200)
#            
#            if var_names == None:
#                var_names = [0,1,2]
#           
#                
#                
#    
#                
#            plt.suptitle('ACFS of ' + str(var_names[0]) + '  ' + str(var_names[1]) + '  ' + str(var_names[2]), fontsize = 14)
#            
#            for col_pi in range(n_cols):
#                #plt.subplot(n_cols, n_cols, (col+1))
#                for i in range(n_cols):
#                    #plt.tight_layout()
#                    
#                    
#                    
#                    
#         
#                    #cL, cH = confint(data_input = data, acf_values = acf_values, n_lags = maxlags)
#                    #x_index = np.arange(0, len(cL)).flatten()
#                    
#                    #if col_pi == i:
#                        
#                    #plt.fill_between(x_index,cL, cH, color='grey', alpha= alpha)
#                    
#                        #lt.fill_between(x_index,cL, cH, color='grey', alpha= alpha)
#            
#                    
#                    #print(i+col+1)
#                    if i== 0:
#                        
#                        norm = max(np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full'))
#                        half = int(np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full').__len__()/2)
#                    
#                        acf_values = np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full')/norm
#                        acf_values = acf_values[half:half + maxlags]
#                        
#                        
#                        
#                        plt.subplot(n_cols,n_cols,i+col_pi +1)
#                        
#                        z = acf_values
#                        cL, cH = confint(data_input = data, acf_values = acf_values, n_lags = maxlags)
#                        x_index = np.arange(0, len(cL)).flatten()
#                  
#                        
#                        plt.fill_between(x_index,cL, cH, color='black', alpha= alpha)
#                        plt.stem(z, linefmt = 'black', markerfmt='ko')
#                        plt.title(var_names[col_pi] + ' ' + var_names[i])
#                      
#                        
#                    if (i== 1) :
#                        
#                        
#                    
#                        #acf_values = np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full')/norm
#                        #acf_values = acf_values[half:half + maxlags]
#                        
#                        plt.subplot(n_cols,n_cols,i+col_pi +3)
#                        
#                        if (col_pi ==0) :
#                            
#                            acf_values = np.correlate(data[:,columns[col_pi]][::-1], data[:,columns[i]], mode ='full')
#                            half = int(len(acf_values)/2)
#                            acf_values = acf_values[half:half + maxlags]
#                            
#                            norm = max(acf_values)
#                            
#                            start = len(acf_values[:half+2])
#                            lead = start - maxlags
#            
#                        
#                            plt.subplot(n_cols,n_cols, i+col_pi +3)
#                            #cL, cH = confint(data_input = data, acf_values = acf_values, n_lags = maxlags, reverse = True)
#                            z = acf_values/norm
#                            cL, cH = confint(data_input = data, acf_values = z, n_lags = maxlags, reverse = True)
#                            
#                            x_index = np.arange(-len(cL)+1, 1).flatten()
#                            
#                            
#                            x = np.arange(-len(z)+1, 1)
#                            plt.stem(x,z, linefmt = 'black', markerfmt='ko')
#                            plt.fill_between(x_index,cL, cH, color='red', alpha= alpha)
#                            plt.title( var_names[col_pi] + ' ' + var_names[i])
#                           
#                            
#                            
#    #                        norm = max(np.correlate(data[:,columns[col_pi]][::-1], data[:,columns[i]], mode ='full'))
#    #                        half = int(np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full').__len__()/2)
#    #                        
#    #                        
#    #                        acf_values = np.correlate(data[:,columns[col_pi]][::-1], data[:,columns[i]], mode ='full')/norm
#    #                        acf_values = acf_values[half:half + maxlags]
#    #                        
#    #                        cL, cH = confint(data_input = data, acf_values = acf_values, n_lags = maxlags, reverse = True)
#    #                        x_index = np.arange(0, len(cL)).flatten()
#    #                        z = acf_values
#    #                        plt.fill_between(x_index,cL, cH, color='red', alpha= alpha)
#    #                        plt.stem(z, linefmt = 'black', markerfmt='ko')
#    #                        plt.title( var_names[col_pi] + ' ' + var_names[i])
#    #                       
#                        else:
#                            
#                            
#                            norm = max(np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full'))
#                            half = int(np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full').__len__()/2)
#                            
#                            acf_values = np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full')/norm
#                            acf_values = acf_values[half:half + maxlags]
#                            
#                            cL, cH = confint(data_input = data, acf_values = acf_values, n_lags = maxlags)
#                            
#                            x_index = np.arange(0, len(cL)).flatten()
#                            z = acf_values
#                            plt.fill_between(x_index,cL, cH, color='black', alpha= alpha)
#                            plt.stem(z, linefmt = 'black', markerfmt='ko')
#                            plt.title( var_names[col_pi] + ' ' + var_names[i])
#                       
#                  
#                        
#    #                    z = acf_values
#    #                    plt.fill_between(x_index,cL, cH, color='grey', alpha= alpha)
#    #                    plt.stem(z, linefmt = 'black', markerfmt='ko')
#    #                    plt.title('ACF ' + ' ' + str(col_pi) + ' ' + str(i))
#    #                
#                   
#                    if (i == 2) and (col_pi != 2):
#                        
#                        
#                    
#                        
#                        
#                        if ((i+col_pi +5) == 7) or ((i+col_pi +5) == 8):
#                            
#                            #norm = np.abs(np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full'))
#                            
#                            #norm = max(norm[len(norm) - maxlags:])
#                            #print('norm is', norm, ' for ', col_pi, i)
#                            #half = int(np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full').__len__()/2)
#                            
#                            
#                            acf_values = np.correlate(data[:,columns[col_pi]][::-1], data[:,columns[i]], mode ='full')
#                            half = int(len(acf_values)/2)
#                            acf_values = acf_values[half:half + maxlags]
#                            
#                            norm = max(acf_values)
#                            
#                            start = len(acf_values[:half+2])
#                            lead = start - maxlags
#            
#                            #acf_values = acf_values[lead:half+1]
#                            
#                            
#                            
#                            #x_index = np.arange(0, len(cL)).flatten()
#                            plt.subplot(n_cols,n_cols, i+col_pi +5)
#                            #cL, cH = confint(data_input = data, acf_values = acf_values, n_lags = maxlags, reverse = True)
#                            z = acf_values/norm
#                            cL, cH = confint(data_input = data, acf_values = z, n_lags = maxlags, reverse = True)
#                            
#                            x_index = np.arange(-len(cL)+1, 1).flatten()
#                            
#                            
#                            x = np.arange(-len(z)+1, 1)
#                            plt.stem(x,z, linefmt = 'black', markerfmt='ko')
#                            plt.fill_between(x_index,cL, cH, color='red', alpha= alpha)
#                            plt.title( var_names[col_pi] + ' ' + var_names[i])
#                        
#                        elif (((i+col_pi +5) != 7) or ((i+col_pi +5) != 8)):
#                            cL, cH = confint(data_input = data, acf_values = acf_values, n_lags = maxlags)
#                            acf_values = np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full')/norm
#                            acf_values = acf_values[half:half + maxlags]
#                            
#                            norm = max(np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full'))
#                            half = int(np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full').__len__()/2)
#                            
#                            x_index = np.arange(0, len(cL)).flatten()
#                            plt.subplot(n_cols,n_cols, i+col_pi +5)
#                            z = acf_values
#                            plt.fill_between(x_index,cL, cH, color='black', alpha= alpha)
#                            plt.stem(z, linefmt = 'black', markerfmt='ko')
#                            plt.title( var_names[col_pi] + ' ' + var_names[i])
#                           
#                        
#                        
#                        #plt.subplot(n_cols,n_cols, i+col_pi +5)
#                        #z = acf_values
#                        #plt.fill_between(x_index,cL, cH, color='grey', alpha= alpha)
#                        #plt.stem(z, linefmt = 'black', markerfmt='ko')
#                        #plt.title('ACF ' + ' ' + str(col_pi) + ' ' + str(i))
#                      
#                        
#                    if (i == 2) and (col_pi == 2):
#                        norm = max(np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full'))
#                        half = int(np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full').__len__()/2)
#                    
#                        acf_values = np.correlate(data[:,columns[col_pi]], data[:,columns[i]], mode ='full')/norm
#                        acf_values = acf_values[half:half + maxlags]
#                        
#                        plt.subplot(n_cols,n_cols, i+col_pi +5)
#                        z = acf_values
#                        #plt.fill_between(x_index,cL, cH, color='black', alpha= alpha)
#                        plt.stem(z, linefmt = 'black', markerfmt='ko')
#                        plt.fill_between(x_index,cL, cH, color='black', alpha= alpha)
#                        plt.title( str(col_pi) + ' ' + str(i))
#                   
#                
#            
#       
        
    
        
       
       
def convert_to_Jordan(data = None, 
                      lags = None, 
                      H = None, 
                      verbose = False, 
                      check = False, 
                      input_matrix = False, 
                      array = None, 
                      zero = True,
                      mode = 'standard'):
    
    from sympy import Matrix
    import numpy as np
    
    def make_blocks(A):
        
        """
        Input a matrix of coefficients and this function will return a block matrix as in Davis and Song 2013
    
        paramters: 
    
            A : numpy array
    
        returns:
    
            BLOCK : numpy array block matrix with the Identity matrix on the diagonals,
            coefficient matricies along the first (block) row and zeros everywhere else
            
        """
        list_of_A = []
        
        for a in range(A.shape[0]):
            list_of_A.append(A[a])
            
        row1 = list_of_A
        p = A.shape[0]
        m = A.shape[1]
        
        if p == 1:
            BLOCK = row1
            
        elif p == 2:
            row2 = [np.eye(m), np.zeros((m,m))]
            BLOCK = np.block([row1, row2])
            
        elif p == 3:
            
            row2 = [np.eye(m), np.zeros((m,m)), np.zeros((m,m))]
            row3 = [np.zeros((m,m)), np.eye(m), np.zeros((m,m))]
            BLOCK = np.block([row1, row2, row3])
        
        return BLOCK
    
    
    
    
    if input_matrix == False:
        if (H == None) or (lags == None):
            print('H and lags must be provided (both positive integer values) ')
            
            
        Mat = GCov_General_phi(fun = GCov_p, args = (data, lags,H, mode), VAR_lags = lags, coefs = 1, resids = 0, acf = 0, eigen = 0, sq = 0, zero = zero)
       
        if lags == 1:
            Mat = Mat[0]
        else:
            Mat = block_coefs(Mat)
            
        Mat = np.array(Mat, dtype = np.float64)
        Mat = np.round(Mat.squeeze(), 10)
        if check:
            print('Mat is', Mat)
            print('The numer of matricies in Mat is ', Mat.shape[0])
    else:
        Mat = array
        
    if lags == 1:
        M     = Matrix(Mat)
        A, J  = M.jordan_form()
        A     = np.array(A, dtype = np.complex)
        A_inv = np.linalg.inv(A)
        J     = np.array(J, dtype = np.complex)
        
        
        
    else:
        import numpy as np
        import scipy as scipy 
        from sympy import Matrix
    
        
        
        U, Q              = scipy.linalg.schur(Mat)
        Q_inv             = np.linalg.inv(Q)
        
        u                 = Matrix(U)
    
        P_tilde, J        = u.jordan_form()
        J                 = np.array(J).astype(np.complex)
        P_tilde_array     = np.array(P_tilde).astype(np.complex)
        P_tilde_array_inv = np.linalg.inv(P_tilde_array)
        
           
        A     = Q.dot(P_tilde_array)
        A_inv = np.linalg.inv(A)
        A     = np.round(A.real, 8)
        J     = np.round(J.real, 8)
        A_inv = np.round(A_inv.real, 8)
        
    if check:
        Mat2  = A.dot(J).dot(A_inv)
        Mat2  = Mat2.real
        return Mat2
    else:
        if verbose:
            print('returns a TUPLE: real matrix M, real matrix A, A inverse and J' )
            print('A and A inverse are ', A.real, A_inv.real)
        return (Mat.real, A.real, A_inv.real, J.real)
        
def make_blocks_external(A, p = None):
        import numpy as np
        
        """
        Input a matrix of coefficients and this function will return a block matrix as in Davis and Song 2013
    
        paramters: 
    
            A : numpy array
            
                array to be turned into a block matrix
            
            p : int
            
                 number of lags
    
        returns:
    
            BLOCK : numpy array block matrix with the Identity matrix on the diagonals,
            coefficient matricies along the first (block) row and zeros everywhere else
            
        """
        list_of_A = []
        
        for a in range(A.shape[0]):
            list_of_A.append(A[a])
            
        row1 = list_of_A
        p = A.shape[0]
        m = A.shape[1]
        
        if p == 1:
            BLOCK = row1
            
        elif p == 2:
            row2 = [np.eye(m), np.zeros((m,m))]
            BLOCK = np.block([row1, row2])
            
        elif p == 3:
            
            row2 = [np.eye(m), np.zeros((m,m)), np.zeros((m,m))]
            row3 = [np.zeros((m,m)), np.eye(m), np.zeros((m,m))]
            BLOCK = np.block([row1, row2, row3])
        
        return BLOCK        
            
            
def NC_irf_external(data = None,
           lags = 1,
           start = 65, 
           end = 85, 
           tol = 100, 
           user_matrix = False,
           user_matrix_data_length = 150,
           array = None, 
           check = False,
           ymax = None,
           ymin = None,
          mod = 'none',
          H = 4,
          verbose = False, 
          zero = True,
          chol = False,
          create_Z = True, graph_margin = 5, save = None, file_path = None, title = "BTC/EHT"):
    """
    
    """
    
    from GCov import convert_to_Jordan
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    print(array)
        
    if user_matrix == False:
        if lags == 1:
            Mat_A_Ainv_J = convert_to_Jordan(data, H = H, zero = zero, lags = lags)# tuple where 0 is the origial matrix, then A, A inverse and J
            
        else:
            raise Exception('this function only does one lag')
        if create_Z:
            Z = GCov_General_phi(fun = GCov_p, args = (data, lags,H, 'standard'), VAR_lags = lags, coefs =0, resids = 1, resids_return = 1, zero = zero)
        #I don't think Z is needed but it can be calculated on demand (switch Z to True to calculate Z)
   
    else:
        Mat_A_Ainv_J = convert_to_Jordan(input_matrix = True, array = array)
     
        if create_Z:
            Z = np.random.randn(user_matrix_data_length,2)
    if create_Z:
        #Z = GCov_General_phi(fun = GCov_p, args = (data, lags,H, 'standard'), VAR_lags = lags, coefs =0, resids = 1, resids_return = 1, zero = zero)
        Σ = np.cov(Z.T)
        PL = np.linalg.cholesky(Σ)
        PL_inv = np.linalg.inv(PL)
    if lags == 1:
        
         J1 = min(np.linalg.eig(Mat_A_Ainv_J[0])[0]) 
         J2 = max(np.linalg.eig(Mat_A_Ainv_J[0])[0])
         print('J1 and J2 are ' , J1, J2)
        
            
    else:
        raise Exception('This function takes on only lag 1')
      

    
        
    A = Mat_A_Ainv_J[1]
    A_inv = Mat_A_Ainv_J[2]

  
   
    Fi_non_neg = np.array([[J1, 0.0],[0.0, 0.0]])
    Fi_neg = np.array([[0.0, 0.0],[0.0, J2]])
    

    
    def F_neg(data, power):
            x = data.copy()
            if power >= 0:
                pass
            else:
                assert power < 0
                J2 = x[1][1]
                J2 = J2**(power)
               
                x[1][1] = J2
            return x
        

    def F_pos(data, power):
        x = data.copy()

        J1 = x[0][0]
        J1 = J1**power
        x[0][0] = J1
        
        return x
    
    list_of_Mi = []
    
    if user_matrix:
        half_length_of_df = int(user_matrix_data_length/2)
        
    else:
        half_length_of_df = 75 #int(len(data)/2)

    if chol:
        for ent in range(-half_length_of_df, half_length_of_df):
            if ent <= -1:
                #ind = ent + half_length_of_df + 1
                temp = A.dot(F_neg(Fi_neg, ent)).dot(A_inv).dot(PL) 
                list_of_Mi.append(temp)
            else:
                temp = A.dot(F_pos(Fi_non_neg, ent)).dot(A_inv).dot(PL)
                list_of_Mi.append(temp)
            
    else:
        for ent in range(-half_length_of_df, half_length_of_df):
            if ent <= -1:
                #ind = ent + half_length_of_df + 1
                list_of_Mi.append(A.dot(F_neg(Fi_neg, ent)).dot(A_inv))
            else:
                list_of_Mi.append(A.dot(F_pos(Fi_non_neg, ent)).dot(A_inv))
            
    if check:
        return list_of_Mi
    else:
        list_of_M1 = []
        for i in range(len(list_of_Mi)):
            list_of_M1.append(list_of_Mi[i][0][0])
        
        list_of_M2 = []
        for i in range(len(list_of_Mi)):
            list_of_M2.append(list_of_Mi[i][0][1])
        
        list_of_M3 = []
        for i in range(len(list_of_Mi)):
            list_of_M3.append(list_of_Mi[i][1][0])
        
        list_of_M4 = []
        for i in range(len(list_of_Mi)):
            list_of_M4.append(list_of_Mi[i][1][1])
        
        M = [list_of_M1, list_of_M2, list_of_M3, list_of_M4]
        M = np.array(M).flatten()
        if ymax == None:
            ymax = max(M) + 0.25*max(M)
        if ymin == None:
            ymin = min(M)  + 0.25*min(M)
            print(ymin)
        
        
        vline_loc = 0
   
        xi = np.arange(-half_length_of_df, half_length_of_df)
        xi = xi[start:end]
       
        plt.figure(figsize = (12,8), dpi = 150)
        plt.tight_layout()

        gap = end - start
        gap = int(gap/2)
        
        plt.suptitle(title + " Noncausal Impulse Response Functions (H = " + str(H) +')', fontsize=16)
        
        plt.subplot(221)
        M1 =pd.Series(list_of_M1)
        
        plt.title('M11')
        plt.plot(xi, M1[start:end])
        #ymin = min(M1[start:end]) - graph_margin
        #ymax = max(M1[start:end]) + graph_margin
        plt.vlines(vline_loc, ymin = ymin, ymax = ymax, alpha = 0.5, color = 'red')
        plt.hlines(vline_loc, xmin = -gap, xmax = gap, alpha = 1, color = 'green')
        plt.ylim(ymin = ymin, ymax = ymax )
   
     
        
    
        plt.subplot(222)
        M2 =pd.Series(list_of_M2)
        
        plt.title('M12')
        plt.plot(xi, M2[start:end])
        #ymin = min(M2[start:end]) - graph_margin
        #ymax = max(M2[start:end]) + graph_margin
        plt.vlines(vline_loc, ymin = ymin, ymax = ymax, alpha = 0.5, color = 'red')
        plt.hlines(vline_loc, xmin = -gap, xmax = gap, alpha = 1, color = 'green')
        plt.ylim(ymin = ymin, ymax = ymax )
       
    
        plt.subplot(223)
        M3 =pd.Series(list_of_M3)
        
        plt.title('M21')
        plt.plot(xi, M3[start:end])
        #ymin = min(M3[start:end]) - graph_margin
        #ymax = max(M3[start:end]) + graph_margin
        plt.vlines(vline_loc, ymin = ymin, ymax = ymax, alpha = 0.5, color = 'red')
        plt.hlines(vline_loc, xmin = -gap, xmax = gap, alpha = 1, color = 'green')
        plt.ylim(ymin = ymin, ymax = ymax )
      
    
        plt.subplot(224)
        
        M4 =pd.Series(list_of_M4)
        
        plt.title('M22')
        plt.plot(xi, M4[start:end])
        #ymin = min(M4[start:end]) - graph_margin
        #ymax = max(M4[start:end]) + graph_margin
        plt.vlines(vline_loc, ymin = ymin, ymax = ymax, alpha = 0.5, color = 'red')
        plt.hlines(vline_loc, xmin = -gap, xmax = gap, alpha = 1, color = 'green')
        plt.ylim(ymin = ymin, ymax = ymax )
        
        if save != None:
            plt.savefig(file_path + save, dpi = 150)
      
def geo_block_sample(data, out = True, print_check = 0, col = None, lags = 1):
        
        import numpy as np
        import pandas as pd 
        import matplotlib.pyplot as plt
        L = pd.DataFrame()
        while len(L)< len(data)+10:
            start = int(np.random.uniform(0, len(data)))
      
            z = np.random.geometric(0.05, len(data))
            length = (z ==1).sum()
            
            if print_check:
                print('start at ' + str(start), 'length of ' + str(length))
            
           
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            sample = data.iloc[start:start + length]
            L = pd.concat([L, sample])
            
                    
            
                
 
                    
        if isinstance(data, pd.DataFrame):
            if len(L) > len(data):
                L = L.iloc[:len(data)]
                
        else:
            if len(L) > len(data):
                L = L[:len(data)]
        
        if out:
            return L.values
        else:
            plt.plot(L.values)
            
            
def mh_boot(data, n_samples = None, lags = 1, H = 11,
                      
                      bells = 2,
           
                  
                      outlier_tol = None,
                      try_zero = True,
                      print_reps_mod = 25,
                      verbose = True,
                      print_iter = True,
                      OLS_boot = False,
                      coef_boot = True, 
                      zero = True,
                      mixed = True):
    
    import pandas as pd
    import numpy as np
    from GCov import GCov_General_phi
    assert lags != None
    assert n_samples != None
    
    
    
    
    
   

    try:
        from playsound import playsound
    except:
        pass
    
  
    from statsmodels.tsa.vector_ar.var_model import VAR as VAR
        
    if outlier_tol == None:
        outlier_tol = 3
        
    
    
    #n_samples = n_samples + 1 #increment number of samples for computational reasons
    
    if verbose:
        print('data type of the input data is', type(data))
        print('calculating resids (once)')
        
    sample_data = geo_block_sample(data)
    sample_data = pd.DataFrame(sample_data)
   

        
    n_cols = 2
    n_cols_sq = n_cols**2
    

    
    
    count = 0
    coefs3 = np.zeros(lags*n_cols_sq)
    while count < n_samples:
        #print(count)
        #if print_iter:
           # print('count is', count)
        if count%print_reps_mod == 0:
            if print_iter:
                
                    print('on iteration ' + str(count) + ' BLOCKS')
                    
        sample_data = geo_block_sample(data)     
        
        if OLS_boot:
            coe = VAR(sample_data ).fit(lags, trend = 'nc').resid
            #print('coe1  OLS is', coe)
        else:
            coe = GCov_General_phi(fun = GCov_p, 
                              args = (sample_data , lags, H), VAR_lags = lags, coefs = 1, resids = 0, resids_return = 1, zero = zero)
        if np.all(np.abs(coe) < outlier_tol):   #print('coe1 GCOV is', coe)
            coe = np.asarray(coe)
            
            row2 = coe.flatten()
           
            coefs3 = np.vstack((coefs3, row2))
            
            count = count + 1
        else:
            count = count
           
            
            
        
   
        
      
             
            
         
    coefs3_df = pd.DataFrame(coefs3)
    #list_of_phi = ['Phi' + str(i) for i in range(1, 4+1)
    list_of_phi = ['Phi1', 'Phi2', 'Phi3', 'Phi4']
        
    coefs3_df.columns = list_of_phi
 
    
 
    
            
    for i in range(bells):
            
        playsound('/Users/Frida/Desktop/A-Tone-His_Self-1266414414.wav')
                
    return coefs3_df.iloc[1:]
 
                  
def VAR_boot_diagnostics(boots, 
                         user_title = "VAR(1) BTC/ETH", 
                         out = False, 
                         file_path = None, 
                         dpi = 150,
                         colors = None, 
                        n_samples = None, diag_mode = 'mean', Φ = None):
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from  scipy.stats import jarque_bera as jb
    import matplotlib.pyplot as plt
    
    if isinstance(boots, list):
        num_phi = len(boots[0].columns)
        
    elif isinstance(boots, pd.DataFrame):
        num_phi = len(boots.columns)
        
    elif isinstance(boots, np.ndarray):
        num_phi = boots.shape[1]
    
    N = n_samples
    B = n_samples
    
    def std_err(boots2, col = None, diag_mode = diag_mode):
        """
        input: boots (list)

        A list of dataframes with bootstraps samples
        """
        if isinstance(boots, list):
            if diag_mode == 'mean':
                list_of_means = []
    
                for i in range(len(boots2)):
                    list_of_means.append(np.mean(boots2[i][col]))
        
                num = np.mean(list_of_means)
                denom = np.std(list_of_means) #*(1/np.sqrt(n_samples))
        
                print('se:       ', denom)
                print('p value:      ', p_value(np.array(list_of_means)))
                print('jb p value:   ', jb(list_of_means)[1])
                print('mean of means:' ,num)
                
            elif diag_mode == 'median':
                print('medians are being used to calculate the standard errors')
                list_of_medians = []
    
                for i in range(len(boots2)):
                    list_of_medians.append(np.median(boots2[i][col]))
        
                num = np.median(list_of_medians)
                denom = np.std(list_of_medians) #*(1/np.sqrt(n_samples))
        
                print('se:       ', denom)
                print('p value:      ', p_value(np.array(list_of_medians)))
                print('jb p value:   ', jb(list_of_medians)[1])
                print('median of medians:' ,num)
                
        
        elif isinstance(boots, pd.DataFrame):
            if diag_mode == 'mean':
                num = np.mean(boots2[col])
                denom = np.std(boots2[col])
                print('se:       ', denom)
                print('p value:      ', p_value(boots2[col]))
                print('jb p value:   ', jb(boots2[col])[1])
                print('mean of means:' ,num)
                
            elif diag_mode == 'median':
                num = np.median(boots2[col])
                denom = np.std(boots2[col])
                print('se:       ', denom)
                print('p value:      ', p_value(boots2[col]))
                print('jb p value:   ', jb(boots2[col])[1])
                print('median of medians:' ,num)
                
    
    for phi in ['Phi' + str(i) for i in range(1,num_phi +1)]:
        print(phi)
        std_err(boots, phi)
        print("        ")
    
    if isinstance(boots, list):
        list_of_list_of_means = []    
        for phi2 in ['Phi' + str(i) for i in range(1,num_phi +1)]:
        
            list_of_means2 = []
    
            for i in range(len(boots)):
                list_of_means2.append(np.median(boots[i][phi2]))
            
            list_of_list_of_means.append(list_of_means2)
        
        plt.figure(figsize = (12,8), dpi = 150)
        if colors == None:
            list_of_colors = ['black', 'grey', 'grey', 'black']
        else:
            if isinstance(colors, list):
                list_of_colors = colors
            else:
                try:
                    list_of_colors = list(colors)
                except:
                    raise Exception('list of colours should be a list')
                    

        for i in range(len(list_of_list_of_means)):
            try:
                plt.subplot(2,2,i+1)
                plt.title(user_title + ' Phi' + str(i+1))
                plt.ylabel('N = B = ' + str(N) )
                sns.distplot(list_of_list_of_means[i], color = list_of_colors[i])
            except:
                pass
        if out:
            try:
                plt.savefig(file_path, dpi = dpi)
            except:
                raise Exception('was not able to save the file')
    
    elif isinstance(boots, pd.DataFrame):
        
        plt.figure(figsize = (12,8), dpi = 150)
        if colors == None:
            list_of_colors = ['black', 'grey', 'grey', 'black']
        else:
            if isinstance(colors, list):
                list_of_colors = colors
            else:
                try:
                    list_of_colors = list(colors)
                except:
                    raise Exception('list of colours should be a list')
                    
        list_of_cols = ['Phi' + str(i) for i in range(1, len(boots.columns)+1)]
        for i in range(len(boots.columns)):
            try:
                plt.subplot(2,2,i+1)
                plt.title(user_title + ' Phi' + str(i+1))
                plt.ylabel( 'B = '  + str(B))
                sns.distplot(boots[list_of_cols[i]], color = list_of_colors[i])
            except:
                pass
        if out:
            try:
                plt.savefig(file_path, dpi = dpi)
            except:
                raise Exception("was not able to save file")
        
    
        
def p_value(data, col = None, tau_hat = 0.0):
    import pandas as pd
    import numpy as np
    
    if isinstance(data, list):
        data = np.array(data)
    
    if isinstance(data, pd.DataFrame):
        data = data.values
        
   

    x1 = -tau_hat
    n1 = len(data)
    y1 = len(data[data <= x1])

  
    z1 = y1/n1
   
    x2 = tau_hat
    n2 = len(data)
    y2 = len(data[data> x2])
    z2 = y2/n2
   
    return 2*min(z1, z2)



def components(data, res = None, H = None, lags = None, CNC_out = None, e_out = None, mode = 'standard', zero = None, e_first = 'e1', e_second = 'e2', Y_first = 'Y1', Y_second = 'Y2', prop_size = 16):
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from GCov import GCov_General_phi, p_value, GCov_p, mh_boot_residuals, convert_to_Jordan, J_n1_n2_external
    
    T = len(data)
   
    temp = GCov_model(data = data, lags =lags ,zero = zero, H =H, mode = mode)
    #res = GCov_General_phi(fun = GCov_p, args = (data, lags, H, mode), VAR_lags = lags, coefs =0, resids = 1, resids_return = 1, acf = 0, eigen = 0, zero = zero, sq = 0)
    T2 = len(res)
    if lags == 1 :
        maaj = convert_to_Jordan(data = data, H = H, mode = mode, lags = lags)
        Asup1 = maaj[2][0]
        Asup2 = maaj[2][1]
        #AsuB1 = maaj[1][:,0].reshape(2,1)
        #AsuB2 = maaj[1][:,1].reshape(2,1)
        n1, n2 = J_n1_n2_external(maaj[3].real)
        print(f'lags is {lags},n1 is {n1} and n2 is {n2}')
        print(maaj[3])
    else:
        
        AJA_inv = temp.AJA_inv
        n1, n2 = J_n1_n2_external(AJA_inv[1].real)
        L_LT1 = []
        L_GT1 = []
        for i in range(AJA_inv[1].shape[0]):
            if np.abs(AJA_inv[1].real[i,i]) < 1:
                L_LT1.append(i)
            else:
                L_GT1.append(i)
        Asup1 = AJA_inv[2][L_LT1,:]
        Asup2 = AJA_inv[2][L_GT1,:]
        
        print(f'lags is {lags},n1 is {n1} and n2 is {n2}')
        
    if n2 == 0:
        raise Exception('Not a mixed process')
    NN = n1 + n2
    p = lags - 1
    causal_colors = ['cyan', 'green', 'grey', 'black', 'magenta', 'yellow']
    ε_colors = ['blue', 'red', 'black', 'green', 'grey', 'magenta', 'yellow']
    try:
        n_cols = len(data.columns)
    except:
        n_cols = data.shape[1]
        
    #causal_and_noncausal = np.empty((T - p, n_cols*lags - n2))
    T = len(data)

    if lags > 0:
        causal_and_noncausal = np.zeros((T,lags*n_cols))
        #causal = np.zeros((T, n1))  
        #noncausal = np.zeros((T, n2))
        
        ε_causal_and_noncausal = np.zeros((T2,lags*n_cols))
        ε_causal = np.zeros((T2, n1))  
        ε_noncausal = np.zeros((T2, n2))
        for i, j in enumerate(range(T - lags)):
            causal_and_noncausal[i, :] =  np.hstack(([data.values[j + k, :] for k in range(lags)]))
        #print('causal and NC: line 2202 ', causal_and_noncausal)
        causal = np.array([Asup1.dot(causal_and_noncausal[i, :]) for i in range(T-lags)])
        
        #LEGACY FOR LOOP
        #for i in range(T - lags):
        
            #try:
            #    causal[i, :] = Asup1.dot(causal_and_noncausal[i, :]) 
            #except:
            #    pass
        temp_NC_list = [Asup2.dot(causal_and_noncausal[i, :]) for i in range(T - lags)]
        noncausal = np.array(temp_NC_list)
        
       
        
        for i, j in enumerate(range(T2 - lags)):
            ε_causal_and_noncausal[i, :] =  np.hstack(([res[j + k, :] for k in range(lags)]))
          

        for i in range(T2 - lags):
          
            try:
                ε_causal[i, :] = Asup1.dot(ε_causal_and_noncausal[i, :]) 
            except:
                print(i, 'ε causal failed')
                print('line 2238')
                pass
            
        for i in range(T2  - lags):
            try:
                ε_noncausal[i, :] = Asup2.dot(ε_causal_and_noncausal[i, :]) 
            except:
                print(i, 'ε noncausal failed')
                print('line 2240')
                pass
        
        causal = np.ma.masked_equal(causal, 0)
        noncausal = np.ma.masked_equal(noncausal, 0)
    
        ε_causal = np.ma.masked_equal(ε_causal, 0)
        ε_noncausal = np.ma.masked_equal(ε_noncausal, 0)
    
    
    
    try:
        CNC = pd.concat([causal, noncausal], axis = 1).dropna()
        #CNC.columns = ['Y1', 'Y2']
    except:
        CNC = pd.concat([pd.DataFrame(causal), pd.DataFrame(noncausal)], axis = 1).dropna()
        CNC.columns = ['Y' + str(i+1) for i in range(len(CNC.columns))]
        
        
    
 
    
    
    if lags >0:
        #LEGACY empty numpy arrays
        #ε_star1 = np.empty((T2,n1))
        #ε_star2 = np.empty((T2,n2))
        
       
        
        print('Asup1.shape is ', Asup1.shape)
        if lags >1:
            #print('NOTE: res is being modifed for n1 = 4 and n2 = 2!')
            z = np.zeros(len(res))
            z.shape = (len(res), 1)
            
            for i in range(2*(lags - 1)):
                res = np.append(res, z, axis = 1)
            #if lags == 2:
             #   res = np.c_[res, z,z]
            #elif lags == 3:
            #    res = np.c_[res, z,z,z,z]
            #elif lags == 4:
              #  res = np.c_[res, z,z,z,z,z,z]
        #LEGACY FOR LOOP
        #for i in range(T2):
        
            #ε_star1[i] = Asup1.dot(res[i, :])
        ε_star1 = np.array([Asup1.dot(res[i, :]) for i in range(T2)])
           
        
        ε_star2 = np.array([Asup2.dot(res[i, :]) for i in range(T2)])
        #LEGACY FOR LOOP
        #for i in range(T2):
            #print('Asup2 is ', Asup2)
            #print('Asup2.dot(res[i, :]).shape is ', Asup2.dot(res[0, :]).shape)
            #ε_star2[i] = Asup2.dot(res[i, :])
        
        
        ε_df = pd.concat([pd.DataFrame(ε_star1), pd.DataFrame(ε_star2)], axis = 1)
        ε_df.columns = ['e' + str(i+1) for i in range(2*lags)]
        #elif lags== 2:
         #   ε_df = pd.concat([pd.DataFrame(ε_star1), pd.DataFrame(ε_star2)], axis = 1)
         #   ε_df.columns = ['e' + str(i) for i in range(1,5)]
        #elif lags == 3:
        #    ε_df = pd.concat([pd.DataFrame(ε_star1), pd.DataFrame(ε_star2)], axis = 1)
        #    ε_df.columns = ['e' + str(i) for i in range(1,7)]
        #elif lags == 4:
        #    ε_df = pd.concat([pd.DataFrame(ε_star1), pd.DataFrame(ε_star2)], axis = 1)
         #   ε_df.columns = ['e' + str(i) for i in range(1,9)]
        #else:
        #    raise ValueError('do lags > 4')
        if e_out == 'plot_separate':
            plt.figure(figsize = (16,12), dpi = 150)
            for i in range(ε_star1.shape[1]):
                num = int(str(ε_star1.shape[1]) + '1' + str(i+1))
                plt.subplot(num)
                
                plt.plot(ε_star1[:, i].astype(float), color = causal_colors[i], linewidth = 1.5)
                plt.hlines(0, xmin = 0, xmax = 150)
          
            for i in range(ε_star2.shape[1]):
                num = int(str(ε_star2.shape[1]) + '1' + str(i+1))
                plt.subplot(num)
                plt.plot(ε_star2[:, i].astyle(float), color = ε_colors[i], linewidth = 1.5)
                plt.hlines(0, xmin = 0, xmax = 150)
      
      
        elif e_out == 'return':
            CNC = None
        
            return ε_df
        
        elif e_out == 'e_shared':
            CNC = None
            cols = data.columns
            #list_of_colors = ['k', 'k']
            plt.figure(figsize = (10,8), dpi = 300)
            for i in range(ε_star1.shape[1]):
                
                plt.plot(ε_star1[:, i].astype(float),linewidth = 2, label = 'ε Causal ' + str(i+1))
            # i in range(ε_star2.shape[1]):
            #    plt.plot(ε_star2[:, i], linestyle = ':', linewidth = 2, label = 'ε Noncausal ' + str(i+1))
                
            plt.legend(prop={'size': prop_size})
         
        elif e_out == 'hist':
            bins = 2*int(np.sqrt(len(ε_star1)))
            CNC = None
            plt.figure(figsize = (16,12), dpi =300)
            plt.subplot(211)
            plt.title('causal ε', fontsize = 14)
            for i in range(ε_star1.shape[1]):
                sns.distplot(ε_star1[:, i].astype(float), color = 'c', bins = bins)
            
            
            plt.subplot(212)
            for i in range(ε_star2.shape[1]):
                sns.distplot(ε_star2[:, i].astype(float), color = 'r', bins = bins)
           
        
        elif e_out == 'joint':
            CNC = None
            #e_out_temp = temp.NC_components(out  = 'e_return')
            e_out_temp = ε_df
            e_temp_1 = e_out_temp.values[:,L_LT1]
            e_temp_1 = e_temp_1[:, 0]
            e_temp_2 = e_out_temp.values[:,L_GT1]
            e_temp_2 = e_temp_2[:, 0]
            
            plt.figure(figsize = (12,8), dpi = 150)
            sns.kdeplot(e_temp_1, e_temp_2, cbar=True)
            plt.title('ε-Causal and ε-Noncausal Joint Plot')
            plt.xlabel('ε-causal')
            plt.ylabel('ε-noncausal')
            #sns.jointplot('e1', 'e2', data = ε_df, kind = 'kde', color = 'green')
        elif e_out == 'e_sep':
            linestyles = [':', '--']
            
            plt.figure(figsize = (16,12), dpi = 300)
     
            for i in range(len(ε_df.columns)):
              
                plot_num = len(ε_df.columns)*100 + 10 + i+1
                plot_num = str(plot_num)
               
                plt.subplot(plot_num)
            
                plt.plot(ε_df['e' + str(i+1)], color = ε_colors[i], linewidth = 1.5, label = 'e' + str(i+1))
               
                plt.hlines(0, xmin = 0, xmax = len(ε_df))
                plt.legend(prop={'size': prop_size})
            
            
        if e_out == None:
            pass
#    elif lags == 2:
#   
#        
#   
#        
#
#        T2p = T2 - p # Tp is 'T less p'
#        begin = T2p/(lags)
#        begin = int(begin) + 1
#
#        ε_causal_and_noncausal = np.zeros((begin*2 -(lags - 1),lags*n_cols))
#
#    
#        ε_causal = np.zeros((T2, n1))  
#        ε_noncausal = np.zeros((T2, n2))
#  
#        
#        for i, j in enumerate(range(begin*2 -(lags - 1) )):
#            ε_causal_and_noncausal[i, :] =  np.hstack(([res[j + k, :] for k in range(lags)]))
#          
#
#        for i in range( T - lags):
#          
#            try:
#                ε_causal[i, :] = Asup1.dot(ε_causal_and_noncausal[i, :]) 
#            except:
#                print(i, 'ε causal failed')
#                pass
#            
#        for i in range( T  -lags):
#            try:
#                ε_noncausal[i, :] = Asup2.dot(ε_causal_and_noncausal[i, :]) 
#            except:
#                print(i, 'ε noncausal failed')
#                pass
            
        #ε_causal = np.ma.masked_equal(causal, 0)
        #ε_noncausal = np.ma.masked_equal(noncausal, 0)
       
          
        ε_df = pd.concat([pd.DataFrame(causal), pd.DataFrame(noncausal)], axis = 1)
        ε_df.columns = ['e' + str(i+1) for i in range(len(ε_df.columns))]
        ε_df = ε_df.dropna()
        linestyles = ['--', ':']  
        
    else:        
        if e_out == 'e_plot':
            
            plt.figure(figsize = (16,12), dpi = 300)
          
            for i in range(len(ε_df.columns)):
                
                #plot_num = len(ε_df.columns)*100 + 10 + i+1
                #plot_num = str(plot_num)
            
                #plt.subplot(plot_num)
                try:
                   plt.plot(ε_df['e' + str(i+1)], color = ε_colors[i], linewidth = 1.5, label = 'e' + str(i+1), linestyle = linestyles[i])
                   plt.hlines(0, xmin = 0, xmax = len(ε_df))
                except:
                   plt.plot(ε_df['e' + str(i+1)], color = ε_colors[i], linewidth = 1.5, label = 'e' + str(i+1), linestyle = linestyles[i])
                   plt.hlines(0, xmin = 0, xmax = len(ε_df))
                
                plt.legend(prop={'size': prop_size})
                
        elif e_out == 'e_sep':
            
            
            plt.figure(figsize = (16,12), dpi = 300)
     
            for i in range(len(ε_df.columns)):
              
                plot_num = len(ε_df.columns)*100 + 10 + i+1
                plot_num = str(plot_num)
               
                plt.subplot(plot_num)
            
                plt.plot(ε_df['e' + str(i+1)], color = ε_colors[i], linewidth = 1.5, label = 'e' + str(i+1), linestyle = linestyles[i])
               
                plt.hlines(0, xmin = 0, xmax = len(ε_df))
                plt.legend(prop={'size': prop_size})
           
                
        elif e_out == 'e_and_original':
            
            plt.figure(figsize = (16,12), dpi = 300)
          
         
            
            plt.plot(ε_df['e1'], color = 'green', linewidth = 1.5, label = 'e1')
            for i in range(len(data.columns)):
                colors_list = ['blue', 'black', 'grey']
                plt.plot(data.values[:, i], color = colors_list[i], linewidth = 1.5, label = 'data')
         
            plt.hlines(0, xmin = 0, xmax = len(ε_df))
            plt.legend(prop={'size': prop_size})
           
            
        
        elif e_out == 'return':
            CNC = None
            #print(type(ε_df))
            return ε_df
        
    
        elif e_out == 'hist':
            if lags == 1:
            
                plt.figure(figsize = (16,12), dpi =300)
                plt.subplot(211)
                plt.title('causal ε', fontsize = 14)
                sns.distplot(ε_star1, color = 'c')
                
                
                plt.subplot(212)
                plt.title('noncausal ε', fontsize = 14)
                sns.distplot(ε_star2, color = 'r')
               
            else:
                plt.figure(figsize = (25,12), dpi = 300)
                for i in range(len(ε_df.columns)):
                    plot_num = len(ε_df.columns)*100 + 10 + i+1
                    plot_num = str(plot_num)
            
                    plt.subplot(plot_num)
                    plt.title(ε_df.columns[i])
               
                    sns.distplot(ε_df['e' + str(i+1)], color = ε_colors[i])
                    plt.hlines(0, xmin = 0, xmax = len(ε_df))
                    plt.legend(prop={'size': prop_size})
                    
        
        elif e_out == 'joint':
            sns.jointplot(e_first, e_second, data = ε_df, kind = 'kde', color = 'green')
    
                
        if e_out == None:
            pass
        
    if CNC_out == 'plot_separate':
        if lags == 1:
            e_out = None
            plt.figure(figsize = (16,12), dpi = 300)
            plt.subplot(211)
            plt.plot(causal, color = 'c', linewidth = 1.5)
            plt.hlines(0, xmin = 0, xmax = 150)
           
            plt.subplot(212)
            plt.plot(noncausal, color = 'r', linewidth = 1.5)
            plt.hlines(0, xmin = 0, xmax = 150)
           
        else:
            raise Exception('lags should equal 1 for this')
        
    elif CNC_out == 'return':
        e_out = None
        return CNC
    
    elif CNC_out == 'all':
        
        
        cols = data.columns
        
        plt.figure(figsize = (10,8), dpi = 300)
        
        
        
        for i in range(NN):
            if i == 0:
                plt.plot(data[cols[0]].values[0:len(data) - lags], color = 'b', linewidth = 0.8, label = 'Original ' + cols[0] +  ' (Detrended)')
                
            if i == 1:
                plt.plot(data[cols[1]].values[0:len(data - lags)], color = 'k', linewidth = 1.0, label = 'Original ' + cols[1] +  ' (Detrended)')
                plt.legend(prop={'size': prop_size})
        
            if i in [i for i in range(n1)]:
                if i + 1 == 1:
                    plt.plot(CNC.values[:,i], color = causal_colors[i], linestyle = '--', linewidth = 1.3, label = 'Causal')
                elif i+1 > 1:
                    plt.plot(CNC.values[:,i], color = causal_colors[i], linestyle = '--', linewidth = 1.3, label = 'Causal ' + str(i+1))
                plt.legend(prop={'size': prop_size})
                
            else:
              
                plt.plot(CNC.values[:,i], color = 'r', linestyle = ':', linewidth = 1.3, label = 'Noncausal')
                plt.legend(prop={'size': prop_size})
        plt.legend(prop={'size': prop_size})

    elif CNC_out == 'original_shared':
        e_out = None
        cols = data.columns
        plt.figure(figsize = (10,8), dpi = 300)
        plt.plot(data[cols[0]].values, color = 'b', linewidth = 0.8, label = 'Original ' + cols[0] +  ' (Detrended)')
        plt.plot(data[cols[1]].values, color = 'k', linewidth = 1.2, label = 'Original ' + cols[1] +  ' (Detrended)')
      
        plt.legend(prop={'size': prop_size})
      
    elif CNC_out == 'CNC_shared':
        e_out = None
        cols = data.columns
        plt.figure(figsize = (10,8), dpi = 300)
       
        
      
        for i in range(NN):
            print('line 2581, NN is ', NN)
            print('CNC.values.shape is ', CNC.values.shape)
            if i in [i for i in range(n1)]:
                print('line 2583 i is ', i)
                if i + 1 == 1:
                    plt.plot(CNC.values[:,i], color = causal_colors[i], linestyle = '--', linewidth = 2, label = 'Causal')
                elif  + 1 > 1:
                    plt.plot(CNC.values[:,i], color = causal_colors[i], linestyle = '--', linewidth = 2, label = 'Causal' + str(i+1))
                plt.legend(prop={'size': prop_size})
                
            else:
                print('reached 2589, the else')
                plt.plot(CNC.values[:,i], color = 'r', linestyle = ':', linewidth = 2, label = 'Noncausal')
              
                plt.legend(prop={'size': prop_size})
                
    
    elif CNC_out == 'joint':
        e_out = None
        if lags == 1:
            plt.figure(figsize = (12,8), dpi = 150)
            
            Y_temp_out = temp.NC_components(out = 'Y_return')
            Y11  = Y_temp_out.values[:,0]
            Y22 = Y_temp_out.values[:,1]
            sns.kdeplot(Y11, Y22, cbar=True)
            plt.title('Y1 and Y2, Causal and Noncausal Components')
            plt.xlabel('Y1')
            plt.ylabel('Y2')
            #sns.jointplot('Y1', 'Y2', data = CNC, kind = 'kde', color = 'black')
        else:
            raise Exception('Have not done for lags > 1, see circa line 2542' )
            #sns.jointplot(Y_first, Y_second, data = CNC, kind = 'kde', color = 'black')
    elif CNC_out == None:
        pass
    
    del temp
    
    
def process_csv(path, date = 'Date', col_name = 'BTC', col_to_convert = None, plot = False):
   
    import pandas as pd
    import matplotlib.pyplot as plt
    def rmv_string(df, x):
        df[x] = df[x].str.replace(r',', r'').astype('float')  
    try:
        df = pd.read_csv(path, parse_dates = [date], index_col = date)
    except:
        print('auto date formatting did not work')
        df = pd.read_csv(path)
        print(df.head())
    
    if col_to_convert == None:
        temp = 'Price'
        rmv_string(df, temp)
    else:
        try:
            c = str(col_to_convert)
            rmv_string(df, c)
        except:
            print('oops, something went wrong')
            print('col_to_convert must be a string')
            
    if col_to_convert == None:
        df = df.Price
    else:
        assert isinstance(col_to_convert, str)
        df = df.col_to_convert

        
    df.columns = col_name
    if plot:
        plt.plot(df)
    else:
        return pd.DataFrame(df)
    
#    
#def CNC_boot(data,resids_input = None, H = 11, lags = 1, mode = 'standard', demean = False, input_matrix = False, array = None):
#    
#    from GCov import GCov_General_phi, p_value, GCov_p, mh_boot_residuals, components, rmv_string, mh_acf, convert_to_Jordan, geo_block_sample, mh_boot
#    import numpy as np
#    import pandas as pd
#    from sklearn.utils import resample
#    
#    if input_matrix:
#
#        maaj = array
#    else:
#        maaj = convert_to_Jordan(data, H = H)
#    
#    
#    Asup1 = maaj[2][0]
#    Asup2 = maaj[2][1]
#    AsuB1 = maaj[1][:,0].reshape(2,1)
#    AsuB2 = maaj[1][:,1].reshape(2,1)
#    
#    if np.all(maaj[3].flatten() >=0):
#        J1 = max(np.abs(maaj[3].flatten()))
#        J2 = min(np.abs(maaj[3].flatten()))  
#    if resids_input == None:       
#        resi = GCov_General_phi(fun = GCov_p, args = (data, lags,H, mode), VAR_lags = lags, coefs =1, resids = 1, acf= 0, resids_return = 1, zero = 1, eigen = 1, sq = 0)
#        resi = resi - np.mean(resi)
#    else:
#        resi = resids_input
#        if demean:
#            resi = resi - np.mean(resi)
#        
#    T = len(resi)
#              
#    re_resi = geo_block_sample(resi)
#    
#    ε_star1 = np.empty(T)
#    ε_star2 = np.empty(T)
#    
#    for i in range(T):
#        ε_star1[i] = Asup1.dot(re_resi[i, :]) 
#    for i in range(T):
#        ε_star2[i] = Asup2.dot(re_resi[i, :])
#            
#    
#    ε = np.concatenate((ε_star1.reshape(1,T), ε_star2.reshape(1,T)), axis = 0)
#    #print(ε.shape)
#    #ε = ε_df.values
#              
#    Y_star1 = np.empty(T)
#    for i in range(T):
#        Y_star1[i] = Asup1.dot(data.values[i, :])
#    
#    
#    Y_star2 = np.empty(T)
#    for i in range(T):
#        Y_star2[i] = Asup2.dot(data.values[i, :])
#    
#    Y_star_df = np.empty_like(ε)
#
#    for t in range(len(ε[0]*2)):
#        
#        if t == 0:
#            Y_star_df[0][0] = 1
#            Y_star_df[1][0] = 1    
#        else:
#            Y_star_df[0][t] = J1*Y_star1[t-1]  + ε[0][t]
#            Y_star_df[1][t] = J2*Y_star1[t-1] +  ε[1][t]
#    #print(Y_star_df.T[:,0])
#    Y_t = AsuB1.dot(Y_star_df.T[:,0].reshape(1,T)) + AsuB2.dot(Y_star_df.T[:,1].reshape(1,T))
#    return Y_t.T
    

def BTC_ETH_import(start = '2018-08-31', 
                   end = '2019-01-27', 
                   plot =False,
                  process = 'demean', scale = 1):
    import pandas as pd
    import matplotlib.pyplot as plt
    from GCov import rmv_string
    '''
    Imports csvs from these specific locations
    
    Parameters:
        start (str) : start date of the selction for both DataFrames
        end (str): end date    
        
        plot (bool): if True then the function plots the data imported but
        returns nothing
        
        process (str): one of 'demean', 'demedian'
        scale (float) : amount to divided BTC by
        
    Returns:
    
        a DataFrame (if plot = False)
    '''
    BTC = pd.read_csv("/Users/Frida/Desktop/Data for Testing Comovements of Exchanges/Crypto including July 2019/BTC_USD Bitfinex up to July 19 2019.csv", parse_dates = ['Date'], index_col = 'Date')
    ETH = pd.read_csv("/Users/Frida/Desktop/Data for Testing Comovements of Exchanges/Crypto including July 2019/ETH_USD Bitfinex Historical Data July 2019.csv", parse_dates = ['Date'], index_col = 'Date')

    rmv_string(BTC, 'Price')
    rmv_string(ETH, 'Price')

    BTC_price = BTC.Price
    ETH_price = ETH.Price

    BTC_price = BTC_price.loc[start:end]
    ETH_price = ETH_price.loc[start:end]
    
    if process == 'demean':
        if scale == 1:
            
            BTC_ETH = pd.concat([BTC_price.sub(BTC_price.mean()).div(scale), ETH_price.sub(ETH_price.mean())], axis = 1)
            BTC_ETH.columns = ['BTC_demeaned', 'ETH_demeaned']
        else:
            BTC_ETH = pd.concat([BTC_price.sub(BTC_price.mean()).div(scale), ETH_price.sub(ETH_price.mean())], axis = 1)
            BTC_ETH.columns = ['BTC_demeaned_div', 'ETH_demeaned_div']
    
    elif process == 'demedian':
        if scale == 1:
            BTC_ETH = pd.concat([BTC_price.sub(BTC_price.median()).div(scale), ETH_price.sub(ETH_price.median())], axis = 1)
            BTC_ETH.columns = ['BTC_med', 'ETH_med']
        else:
            BTC_ETH = pd.concat([BTC_price.sub(BTC_price.median()).div(scale), ETH_price.sub(ETH_price.median())], axis = 1)
            BTC_ETH.columns = ['BTC_med_div', 'ETH_med_div']
    else:
        if scale ==1 :
            BTC_ETH = pd.concat([BTC_price.div(scale), ETH_price], axis = 1)
            BTC_ETH.columns = ['BTC', 'ETH']
        else:
            BTC_ETH = pd.concat([BTC_price.div(scale), ETH_price], axis = 1)
            BTC_ETH.columns = ['BTC_div', 'ETH_div']
    
    
    if plot:
        plt.plot(BTC_ETH)
    else:
        return BTC_ETH
    
    
def estimate(data, H = 4, lags = 1, zero =1, display = True):
    import numpy as np

    from statsmodels.graphics.gofplots import qqplot as qq
    import matplotlib.pyplot as plt
    from GCov import GCov_General_phi, GCov_p
    import seaborn as sns
    from  scipy.stats import jarque_bera as jb
    from scipy.stats import kstest as ks
    from scipy.stats import wilcoxon as cox
    
    from numpy.random import standard_t as t

    
    # list of coefs
    c = GCov_General_phi(fun = GCov_p,  args = (data, lags, H), VAR_lags = lags, coefs = 1, resids = 0, acf = 1, resids_return = 0, zero = zero, sq = 0, eigen = 0)
    
    e = GCov_General_phi(fun = GCov_p,  args = (data, lags, H), VAR_lags = lags, coefs = 1, resids = 0, acf = 1, resids_return = 0, zero = 1, sq = 0, eigen = 1)
    
    dw = GCov_General_phi(fun = GCov_p,  args = (data, lags, H), VAR_lags = lags, coefs = 1, resids = 1, acf = 0, resids_return = 0, zero = 1, sq = 0, eigen = 0, durbin_watson  = 1)
    nor = np.random.randn(150)
    nor2 = np.random.randn(9999)
    if display:
        plt.figure(figsize = (12,8), dpi = 150)
        cauchy = np.random.standard_cauchy(150)
        res = GCov_General_phi(fun = GCov_p,  args = (data, lags, H), VAR_lags = lags, coefs = 1, resids = 1, acf = 1, resids_return = 1, zero = zero, sq = 0, eigen = 0)
        plt.subplot(611)
        sns.distplot(res[:,0])
        plt.subplot(612)
        sns.distplot(res[:,1])
        plt.subplot(613)
        sns.distplot(nor)
        plt.subplot(614)
        sns.distplot(nor2)
        plt.subplot(615)
        sns.distplot(cauchy)
        ttt = t(4,150)
        plt.subplot(616)
        sns.distplot(ttt)
        
        plt.subplots()
        qq(res[:,0],line = '45')
        
        qq(res[:,1], line = '45')
        
        qq(nor, line = '45')
        
        qq(cauchy, line = '45')
        qq(ttt, line = '45')
        print('coefficients: ', c)
        print('Eigenvalues: ', e)
        print('dw stat: ', dw)
        print('JB for residuals 2: ', jb(res[:,0]))
        print('JB for residuals 2: ', jb(res[:,1]))
        print('Kolmogorov-Smirnov 1:', ks(res[:,0], 'norm'))
        print('Kolmogorov-Smirnov 2:', ks(res[:,1], 'norm'))
        print('Wilcoxon 1:', cox(res[:,0]))
        print('Wilcoxon 2:', cox(res[:,1]))
    else:
        return (c,e,dw)
    
    
def spline_detrend(data, order = 2, plot = True, obs_between_knots = 50, ret_values = True, col_names = None):
    
    from obspy.signal.detrend import spline as spline
    import pandas as pd
   
    if col_names == None:
        col_names = ['BTC', 'ETH']
        
    if isinstance(data, pd.DataFrame):
        df = data.copy() 
        df = df.values
       
        out1 = spline(df[:,0], order=order, dspline=obs_between_knots, plot= plot)
        out2 = spline(df[:,1], order=order, dspline=obs_between_knots, plot= plot)
        if ret_values:
            df_out = pd.concat([pd.Series(out1), pd.Series(out2)], axis = 1)
            df_out.columns = [col_names[0] + '_det', col_names[1] + '_det']
            return df_out
        
        
        
        
def mboot(data, diag_mode = 'mean', d = True, lags = 1, H= 4, B = 10, outlier_tol = None, diag_only = False, verbose = 1, print_iter = 1, OLS_boot = False, mixed = True, zero = True):
    
    import numpy as np
    from GCov import mh_boot_residuals, VAR_boot_diagnostics
    list_of_boots= []
    
        
    if outlier_tol == None:
        
        for i in range(B):
            phis = mh_boot_residuals(data, lags = 1, n_samples = B, bells = 0, outlier_tol = np.inf, H = H, verbose = verbose, print_iter = print_iter, OLS_boot = OLS_boot, zero = zero, mixed = mixed, diag = 0)
            list_of_boots.append(phis)
            print(i+1, 'of ', B)
       
    else:
        assert outlier_tol > 0
        for i in range(B):
            phis = mh_boot_residuals(data, lags = 1, n_samples = B, bells = 0, outlier_tol = outlier_tol, H = H, verbose = verbose, print_iter = print_iter, OLS_boot = OLS_boot, zero = zero, mixed = mixed, diag = 0)
            list_of_boots.append(phis)
            print(i+1, 'of ', B)
            
    if d:
        VAR_boot_diagnostics(list_of_boots, n_samples = B, diag_mode = diag_mode)
    else:
        return list_of_boots
 

#def thesis_latex(data = None, 
#                   H = None, 
#                   lags_list_range = 3,
#                   txt_file_path = None, 
#                   jpg_file_path = None, 
#                   round_off = 6):
#    import numpy as np
#    def the_normal_function(data, first = None, second = None, verbose = False, ks = None, jb = None, watts = None):
#        import pandas as pd
#        
#        try:
#            n_cols = len(data.columns)
#        except:
#            data = pd.DataFrame(data)
#            n_cols = len(data.columns)
#    
#        variables = [ i for i in data.columns]
#        
#        if ks:
#            from statsmodels.stats.diagnostic import kstest_normal as ks
#            
#            if verbose:
#                print('this is the kstest normality test with estimated mean and var')
#                print('the first entry is the statistic, the second is the p-value')
#                print('if neither first nor second is True then all four values are returned')
#            
#            
#            if first:
#                first_var_statistic = ks(data[variables[0]])[0]
#                first_var_p = ks(data[variables[0]])[1]
#                temp = [first_var_statistic, first_var_p]
#                return temp
#            
#            elif second:
#                second_var_statistic = ks(data[variables[1]])[0]
#                second_var_p = ks(data[variables[1]])[1]
#                temp = [second_var_statistic, second_var_p]
#                return temp
#            
#            else:
#                first_var_statistic = ks(data[variables[0]])[0]
#                first_var_p = ks(data[variables[0]])[1]
#                second_var_statistic = ks(data[variables[1]])[0]
#                second_var_p = ks(data[variables[1]])[1]
#                temp = [first_var_statistic, first_var_p, second_var_statistic, second_var_p]
#                return temp
#        if jb:
#            from  scipy.stats import jarque_bera as jb
#            if first:
#                first_var_statistic = jb(data[variables[0]])[0]
#                first_var_p = jb(data[variables[0]])[1]
#                temp = [first_var_statistic, first_var_p]
#                return temp
#            
#            elif second:
#                second_var_statistic = jb(data[variables[1]])[0]
#                second_var_p = jb(data[variables[1]])[1]
#                temp = [second_var_statistic, second_var_p]
#                return temp
#            
#            else:
#                first_var_statistic = jb(data[variables[0]])[0]
#                first_var_p = jb(data[variables[0]])[1]
#                second_var_statistic = jb(data[variables[1]])[0]
#                second_var_p = jb(data[variables[1]])[1]
#                temp = [first_var_statistic, first_var_p, second_var_statistic, second_var_p]
#                return temp
#            
#    f = open(txt_file_path, "w+")
#    
#    #lags_list = [(i+1) for i in range(lags_list_range)]
#    f.write('\n')
#    f.write('Data columns are ' + str(data.columns[0]) + ' and '  + str(data.columns[1]) + '\n')
#    space = '=================================== \n'
#    f.write(space)
#    for lag in [1,2,3]:
#        c = GCov_General_phi(args = (data, lag, H),VAR_lags = lag, resids = 0, coefs = 1)
#        
#        c = np.asarray(c)
#        #c = np.around(c, decimals = round_off)
#        
#        r = GCov_General_phi(args = (data, lag, H),VAR_lags = lag, resids = 1, resids_return = 1, coefs = 0)
#        
#        if lag == 1:
#            c = c.flatten()
#            
#            c = tuple(c)
#            
#            f.write(space)
#            f.write('Coefficients for Lag = ' + str(lag)  + ' (Where H = ' + str(H*lag))
#            f.write('\n')
#            f.write('\n')
#            f.write('this is NEW:')
#            f.write('\n')
#            f.write('\n')
#            f.write('$$\[ \Phi_{GCOV_{BTC/ETH}}=\left[ {\\begin{array}{cc} \n %f & %f \\\\ \n %f & %f\\\\ \n \\end{array} } \\right] \]$$ '% c)
#                    
#                    #\n MATRIX 1 \n \n \n \\[\\begin{bmatrix} \n %f & %f \\\\ \n %f & %f\\\\ \n \\end\{bmatrix\}\\]'% c)
#            
#            f.write('\n')
#            f.write('\n')
#        
#        
#        if lag == 2:
#            #c2 = GCov_General_phi(args = (data, lag, H*lag),VAR_lags = lag, resids = 0, coefs = 1)
#            #c2 = np.asarray(c2)
#            #c2 = np.around(c2, decimals = round_off)
#            c2_1 = tuple(c.flatten()[:4])
#            c2_2 = tuple(c.flatten()[4:])
#            f.write(space)
#            f.write(space)
#            f.write('\n')
#            #f.write(space)
#            f.write('Coefficients for Lag = ' + str(lag))
#            f.write(' (Where H = ' + str(H)  + ') \n')
#            f.write(' \n MATRIX 1 \n')
#            f.write('\n')
#            f.write('\n')
#            f.write('$$\[ \Phi_{GCOV_{BTC/ETH}}=\left[ {\\begin{array}{cc} \n %f & %f \\\\ \n %f & %f\\\\ \n \\end{array} } \\right] \]$$ '% c2_1)
#            f.write('\n')
#            f.write('\n')
#            
#        
#            f.write(' \n MATRIX 2 \n')
#            f.write('\n')
#            f.write('$$\[ \Phi_{GCOV_{BTC/ETH}}=\left[ {\\begin{array}{cc} \n %f & %f \\\\ \n %f & %f\\\\ \n \\end{array} } \\right] \]$$ '% c2_2)
#            f.write('\n')
#            f.write('\n')
#            
#            
#            
#        if lag == 3:
#            #c3 = GCov_General_phi(args = (data, lag, H*lag),VAR_lags = lag, resids = 0, coefs = 1)
#            #c3 = np.asarray(c3)
#            #c3 = np.around(c3, decimals = round_off)
#            c3_1 = tuple(c.flatten()[:4])
#            c3_2 = tuple(c.flatten()[4:8])
#            c3_3 = tuple(c.flatten()[8:])
#            #f.write(space)
#         
#            f.write('\n')
#            f.write('Coefficients for Lag = ' + str(lag))
#            f.write(' (Where H = ' + str(H)  + ') \n')
#            f.write(' \n MATRIX 1 \n')
#            f.write('\n')
#            f.write('\n')
#            #f.write('\\[\\begin{bmatrix} \n %f & %f \\\\ \n %f & %f\\\\ \n \\end\{bmatrix\}\\]'% c3_1)
#            f.write('$$\[ \Phi_{GCOV_{BTC/ETH}}=\left[ {\\begin{array}{cc} \n %f & %f \\\\ \n %f & %f\\\\ \n \\end{array} } \\right] \]$$ '% c3_1)
#            f.write('\n')
#            f.write('\n')
# 
#            
#            f.write(' \n MATRIX 2 \n')
#            f.write('\n')
#            f.write('$$\[ \Phi_{GCOV_{BTC/ETH}}=\left[ {\\begin{array}{cc} \n %f & %f \\\\ \n %f & %f\\\\ \n \\end{array} } \\right] \]$$ '% c3_2)
#            f.write('\n')
#            f.write('\n')
#            
#            f.write(' \n MATRIX 3 \n')
#            f.write('\n')
#            f.write('$$\[ \Phi_{GCOV_{BTC/ETH}}=\left[ {\\begin{array}{cc} \n %f & %f \\\\ \n %f & %f\\\\ \n \\end{array} } \\right] \]$$ '% c3_3)
#            f.write('\n')
#            f.write('\n')
#            
#        ##### KS NORMALITY TESTS #######
#        #f.write(space)
#        
#        KS_1_data1 = the_normal_function(data, first= 1, ks = 1)
#        KS_1_data2 = the_normal_function(data, second= 1, ks = 1) 
#            
#        KS_1_var1 = the_normal_function(r, first= 1, ks = 1)
#        KS_1_var2 = the_normal_function(r, second = 1, ks = 1)
#        
#        JB_1_data1 = the_normal_function(data, first= 1, jb = 1)
#        JB_1_data2 = the_normal_function(data, second= 1, jb = 1) 
#            
#        #JB_1_var1 = the_normal_function(r, first= 1, jb = 1)
#        #JB_1_var2 = the_normal_function(r, second = 1, jb = 1)
#        
#        f.write(space)
#        f.write('\n')
#        f.write('\n')
#        f.write('The JB test statistic on the data itself (BTC) is ' + str(round(JB_1_data1[0],4)))
#        f.write('\n')
#        f.write('\n')
#        
#        f.write(space)
#        f.write('\n')
#        f.write('\n')
#        f.write('The JB test statistic on the data itself (BTC) is ' + str(round(JB_1_data2[0],4)))
#        f.write('\n')
#        f.write('\n')
#        
#        f.write(space)
#        f.write('\n')
#        f.write('\n')
#        f.write('The Kolmogorov-Smirnov test statistic on the data itself (BTC) is ' + str(round(KS_1_data1[0],4)))
#        f.write('\n')
#        f.write('\n')
#        
#        f.write(space)
#        f.write('\n')
#        f.write('\n')
#        f.write('The Kolmogorov-Smirnov test statistic on the data itself (ETH) is ' + str(round(KS_1_data2[0],4)))
#        f.write('\n')
#        f.write('\n')
#        
#        
#        f.write('\n')
#        f.write('\n')
#        f.write('The Kolmogorov-Smirnov test statistic on residuals with estimated mean and variance \n at lag ' + str(lag))
#        f.write('\n')
#        f.write('\n')
#           
#        f.write('\n')
#        f.write('At Lag ' + str(lag) + ' the statistic value on residuals is ' + str(round(KS_1_var1[0],4)) + ' for ' + data.columns[0])
#        f.write('\n')
#        f.write('At Lag ' + str(lag) + ' the statistic value on residuals is ' + str(round(KS_1_var2[0],4)) + ' for ' + data.columns[1])
#        f.write('\n')
#        f.write('\n')
#        f.write('The statistic value is ' + str(round(KS_1_var1[1],4)) + ' for ' + data.columns[0])
#        f.write('\n')
#        f.write('The statistic value is ' + str(round(KS_1_var2[1],4)) + ' for ' + data.columns[1])
#        f.write('\n')
#        f.write('\n')
#        f.write(space)
#            
            
        
        
class GCov_model:
    
    import GCov
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.tsa.vector_ar.var_model import VAR as VAR
    from  scipy.stats import jarque_bera as jb
    from statsmodels.regression.linear_model import OLS as OLS
    from statsmodels.stats.stattools import durbin_watson as dw
    
  
    from statsmodels.tsa.stattools import adfuller as ad
    from statsmodels.tsa.arima_model import ARIMA as ARIMA
    from scipy.stats import kurtosis as kurt
    from scipy.stats import skew as skew
    from statsmodels.stats.diagnostic import kstest_normal as ks
    from scipy.stats import wilcoxon as wilcox
    from statsmodels.graphics.gofplots import qqplot as qq
    import importlib
    from scipy.stats import normaltest as normaltest
    from scipy.stats import shapiro as shapiro
    from scipy.stats import anderson as anderson
   
    def __init__(self, data, lags, H, mode = 'standard', verbose = False, file_path = None, zero = 1):
        
        import numpy as np
        """
        Help on the GCov_model class: 
            
        __init__(self, data, lags, H, mode = 'standard', save = None, verbose = False):
         
         This class contains the original data to be analysed, residuals, squared residuals, dw statistics,
         a block matrix of coefficients, a regular matrix of coefficients and eigen values for the model.
         
         The class can can also output ACFs of the data, the residuals, the squared residuals, 
         the causal and noncausal components. 
         
         The class can also output graphs of the joint distribution of the data, the causal and noncausal errors, and the
         causal and noncausal components of the data. 
         
         The class can output graphs of the noncausal Impuse Response Function for the data. 
         
         The class can perform bootstrap estimation on the paramters of the model and calcualte an
         estimated p-value for the parameters of the model. 
        
    
        Parameters
        ----------
        
        data: nd data array
            The number of columns = the number of time series in the data
            
        mode: string
            changes the objective function of the GCov_model class
            
        lags : int
            A number of lags must be given to instantiate the model object.
       
        verbose : bool
            If True the model will print statments as it runs at various points.  
            
        
        
        Returns
        -------
        fGCov_model object
        
    
    
    
        Examples
        --------
        >>> import GCov
        >>> model = GCov_model(data, lags = 1, H = 4) # instantiate a model of any lags and any H
        >>> model.coeffs                              # outputs an nd array of autoregressive coefficients estimated by GCov
        >>> model.eigen                               # outputs a list of lists of eigen values (one list for each lag)
        >>> model.acf()                               # outputs a graph of ACFs of the data given at instantiation
        >>> model.acf_sq()                            # outputs an ACF of the squared data (not really needed to be honest)
        >>> model.res_acf()                           # outputs an ACF of the residuals of the model
        >>> model.res_acf_sq()                        # outputs an ACF of the squared residuals
        >>> model.boot(B = 1000)                      # performs a bootstrap of the parameters of the model, default B is 300
        >>> model.NC_irf()                            # outputs a graph of the Noncausal Impuse Response Function]
        >>> model.hist()                              # outputs histograms of the data
        >>> model.res_hist()                          # outputs histograms of the residuals of the data
        >>> model.NC_components(out = 'e-plot')       # plots the causal and noncausal errors of the model
        >>> model.NC_components(out = 'e-joint')      # plots a joint distribution of the causal and noncausal errors of the model
        >>> model.NC_components(out = 'Y_plot')       # plots the causal and Noncausal components of the data
        >>> model.NC_components(out = 'plot_all')     # (for 1 lag only) plots the causal and noncausal errors and components of the model
        >>> model.NC_components(out = 'e-return')     # returns the causal and noncausal errors of the process
        >>> model.NC_components(out = 'Y_return')     # returns the causal and noncausal components of the process
        >>> model.NC_components(out = 'e_acf')        # returns an ACF of the causal and noncausal errors of the process

        >>> model.qq_plot()                           # outputs a QQ plot of either the data of the model or the residuals of the model
        >>> model.joint()                             # outputs a graph of the joint distribution of the data (assumes bivariate data)
        >>> model.res_joint()                         # outputs a graph of the joint distribution of the residuals of the model
        >>> model.stats()                             # outputs a pandas dataframe of a range of statistics on the data or the residuals
        
        """
        
        
        from statsmodels.stats.stattools import durbin_watson as dw
        if file_path != None:
            self.file_path = file_path
        self.data = data
        try:
            self.n_cols = len(data.columns)
        except:
            self.n_cols = data.shape[1]
        
        
        self.lags = lags
        self.H = H
        self.mode = mode
        self.coefs = GCov_General_phi(fun = GCov_p,  args = (self.data, self.lags, self.H, self.mode), VAR_lags = self.lags, coefs = True, std_err = False, resids = 0, acf = 0, resids_return = 0, zero = zero, sq = 0, eigen = 0)
       
        self.zero = zero
        
        self.se =  GCov_General_phi(fun = GCov_p,  args = (self.data, self.lags, self.H, self.mode), VAR_lags = self.lags, std_err = 1, coefs = 0, resids = 0, acf = 0, resids_return = 0, zero = zero, sq = 0, eigen = 0)
        self.res =  GCov_General_phi(fun = GCov_p,  args = (self.data, self.lags, self.H, self.mode), VAR_lags = self.lags, coefs = 1, resids = 1, acf = 0, resids_return = 1, zero = zero, sq = 0, eigen = 0)
        self.res_sq = self.res*self.res
     
        self.eigen = np.linalg.eig(self.coefs)[0]
        self.dw = dw(self.res)
        self.dw_sq = dw(self.res_sq)
        if lags == 1:
            self.wald = self.coefs[0].flatten()/self.se
        else:
            self.wald = self.coefs.flatten()/self.se
            
        if lags > 1:
            try:
                self.block_coefs = block_coefs(self.coefs)
                self.J = block_Jordan(self.block_coefs, n_cols = self.n_cols)
            except:
                
                print('converting to block matrix of coefficients did not work')
        if verbose:
            print('eigenvalues for ' + str(self.lags) + ' lags & ' + 'H = ' + str(self.H) + ': ', self.eigen)
        else:
            pass
        if lags > 1:
            self.AJA_inv = find_A_external2(self.block_coefs)
        else:
            self.AJA_inv = find_A_external2(self.coefs)
        
    def find_mode(self, lags = None, H = None, acf= 'NC_components', verbose = False):
     
    
        data = self.data
        if lags == None:
            lags = self.lags
        if H == None:
            H = self.H
            
        for i, mode in enumerate(['standard', 'exp', 'factorial']):
            model = GCov_model(data, lags, H, mode = mode, verbose = False)
   
            if acf == 'NC_components':
                if verbose:
                    print('NC_components ACF ' + mode)
                model.NC_components(out = 'e_acf', mode = mode, verbose = False)
                
            elif acf == 'res':
                if verbose:
                    print('Residuals ACF ' + mode)
                model.res_acf()
                
        del model
        

    def findH(self, num_lags = 3, num_H = 20, zero = True, sq = False):
        
        """
        Help on function finH in module GCov_model:

         findH(self, num_lags = 3, num_H = 20, zero = True, sq = False)
         
         This function graphs, for each lag equal to or less than num_lags
         and each H equal to or less then num_H,  the absolute value of the sum of absolute values
         of the difference between a calcualted Durbin-Watson statistic  and 2 ( where the Durbin-Watson
         statistic isAPPLIED TO THE RESIDUALS OF THE GCOV MODEL and where
         a value of 2 indicates a lack of serial correlation in the data. 
        
    
        Parameters
        ----------
        
            nd data array
        num_lags : int
            The number of lags to use for comparison, default is 3.
       
        num_H : int
            The number of lags H to use for each lag. 
            
        zero : bool
            If True then the starting values used are the zero vector of appropriate dimension
            If False then the GCov_model uses OLS starting values. 
            
        sq : bool
            If true then the Durbin-Watson statistic is calculated on the squared residuals
        
        Returns
        -------
        fig : Matplotlib figure instance
        
    
    
    
        Examples
        --------
        >>> import GCov
        >>> model = GCov_model(data, lags = 1, H = 4) # instantiate a model of any lags and any H
        >>> model.findH() # outputs a graph for 3 lags over values of H from 2 to 20
        
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        dw_list1 = []
        dw_list2 = []
        dw_list3 = []
        if num_lags == 4:
            dw_list4 = []
        
        if sq:
            for i in range(1,num_lags + 1):
                for h in range(2,num_H):
                    d =  GCov_General_phi(fun = GCov_p,  args = (self.data, i, h), VAR_lags = i, coefs = 1, resids = 1, acf = 0, resids_return = 0, zero = zero, sq = 1, eigen = 0, durbin_watson = 1)
                    if i == 1:
                        d1 = abs(abs(d[0]- 2) + abs(d[1]-2))
                        dw_list1.append(d1)
                    elif i == 2:
                        d2 = abs(abs(d[0]- 2) + abs(d[1]-2))
                        dw_list2.append(d2)
                    elif i == 3:
                        d3 = abs(abs(d[0]- 2) + abs(d[1]-2))
                        dw_list3.append(d3)
                    if num_lags ==4:
                        if i == 4: 
                            d4 = abs(abs(d[0]- 2) + abs(d[1]-2))
                            dw_list4.append(d4)
                    
        else:
            for i in range(1,num_lags+1):
                for h in range(2,num_H):
                    d =  GCov_General_phi(fun = GCov_p,  args = (self.data, i, h), VAR_lags = i, coefs = 1, resids = 1, acf = 0, resids_return = 0, zero = zero, sq = 0, eigen = 0, durbin_watson = 1)
                    if i == 1:
                        d1 = abs(abs(d[0]- 2) + abs(d[1]-2))
                        dw_list1.append(d1)
                    elif i == 2:
                        d2 = abs(abs(d[0]- 2) + abs(d[1]-2))
                        dw_list2.append(d2)
                    elif i == 3:
                        d3 = abs(abs(d[0]- 2) + abs(d[1]-2))
                        dw_list3.append(d3)
                    if num_lags ==4:
                        if i == 4: 
                            d4 = abs(abs(d[0]- 2) + abs(d[1]-2))
                            dw_list4.append(d4)
        plt.figure(figsize = (12,8), dpi = 150)         
            
        plt.plot(dw_list1, color = 'red', label = 'lag1')

        plt.plot(dw_list2, color = 'blue', label = 'lag2')

        plt.plot(dw_list3, color = 'green', label = 'lag3')
        if num_lags == 4:
            plt.plot(dw_list4, color = 'black', label = 'lag4')
        plt.xlabel('H - lags')
        
        #### TODO #### Fix the labels
        #if sq:
            #plt.title('Euclidian distance of the deviations of the Durbin-Watson statistic from 2 for SQUARED residuals')
        #else:
            #plt.title('Euclidian distance of the deviations of the Durbin-Watson statistic from 2 for residuals')
        #plt.legend()
        
    ### PLOTS ###
    
    def qq_plot(self, res_plot = True, verbose = True, QQ_color = None, QQ_type = None, save = None):
        """
        Outputs a graph either of the data used in the GCov_model class or the residuals from said model
        
        Parameters:
        res_plot (bool): 
            Default False, if true the QQ plot of the residuals of the model are output
                otherwise a QQ plot of the data used in estimation is output
                
        verbose (bool): 
            if true extra information is printed when the functino is called
        
        QQ_color (str) : 
            sets the color of the QQ plot if given
        
        QQ_type (str):  Default '45' must be one of {'45', 's', 'r', 'q'} 
            (The default is None at the input but if left as None then '45' is chosen)
       
            Options for the reference line to which the data is compared:
    
                - '45' - 45-degree line
                - 's' - standardized line, the expected order statistics are scaled
                    by the standard deviation of the given sample and have the mean
                    added to them
                - 'r' - A regression line is fit
                - 'q' - A line is fit through the quartiles.
                - None - by default no reference line is added to the plot.
        save (str): 
            if a filepath is given then the qq plot is saved to this location
        Returns:
            
        A graph of the QQ plots, if save is not None it attempts to save the file following the filepath 
            given in the save argument
        """
        from statsmodels.graphics.gofplots import qqplot as qq
        import matplotlib.pyplot as plt
        data = self.data
        res = self.res
        try:
            n_cols = len(data.columns)
        except:
            n_cols = data.shape[1]
        plt.figure(figsize = (12,8), dpi = 150)
        for i in range(n_cols):
            plot_num = n_cols*100 + 11 + i
            plot_num = str(plot_num)
            
            if verbose:
                print('plot number for subplt is ', plot_num)
            
            if res_plot:
                # convert to a numpy array if res is not already
                try: 
                    res = res.values
                except:
                    pass
                plt.subplot(plot_num)
               
                if QQ_color == None:
                    
                    qq(res[:,i], line  = 'q')
                else:
                    qq(res[:,i], line  = 'q', color = QQ_color)
            else:
                 #convert to a numpy array if data is not already
                try:
                    data = data.values
                except:
                    pass
                plt.subplot(plot_num)
            
                if QQ_color == None:
                    qq(data[:,i], line = '45')
                else:
                    qq(data[:,i], line = '45', color = QQ_color)
        if save == None:
            pass
        else:
            assert isinstance(save, str), 'save must be a string'
            try:
                plt.savefig(save)
            except:
                raise Exception('Was not able to save the file')
                
        
    def joint(self,var1 = None, var2 = None):
        """
        Outputs a graph of the joint distribution of the data used in the class (not the residuals)
        """
        import seaborn as sns
        if (var1 == None) and (var2 == None):
            sns.jointplot(self.data.columns[0], self.data.columns[1], data = self.data, kind = 'kde')
        else:
            #For data that has is larger than bivariate
            assert isinstance(var1, int)
            assert isinstance(var2, int)
            sns.jointplot(self.data.columns[var1], self.data.columns[var2], data = self.data, kind = 'kde')
        
    def hist(self, colors = None, save = None, var_names = None):
        """
        Help for the hist method in the GCov model class
        
        hist(self, colors = None)
        
        
        Outputs a histogram of the data used in the class (not the residuals)
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        n_cols = len(self.data.columns)
        n1 = n_cols*100
        n2 = 10
        n = n_cols + 1
        plt.figure(figsize = (12, 8))
        
        for i in range(1, n):
            ind = n1+ n2 + i
            ind = str(ind)
          
            j = i-1
            plt.subplot(ind)
            plt.title(self.data.columns[j])
            if colors == None:
                
                sns.distplot(self.data.values[:, j])
            else:
                assert isinstance(colors, list)
                assert len(colors) == len(self.data.columns)
             
                sns.distplot(self.data.values[:, j], color = colors[j])
                if var_names != None:
                    plt.title(var_names[i-1])
        if save != None:
            file_path = self.file_path
            plt.savefig(file_path + save, dpi = 150)
    def stats(self, res = False):
        """
        
        This function outputs a datagram of a several statistics on given data including:
            
        Mean
        Standard Deviation
        Kurtosis
        Skew 
        Minimum
        Maximum
        Jarque - Bera
        Wilcoxon
        Komolgorov-Smirnoff
        Normaltest
        Shapiro
        Anderson Darling
        Augmented Dickey-Fuller with no constant, constant, constant and linear trend 
            and constant linear trend and quad trend (nc, c, ct and ctt respectively)
               
        """
        
        import numpy as np
        import pandas as pd
       
        
        
        
        from scipy.stats import kurtosis as kurt
        from scipy.stats import skew as skew
        from  scipy.stats import jarque_bera as jb
        from scipy.stats import wilcoxon as wilcox
        from scipy.stats import kstest as ks
        from scipy.stats import normaltest as normaltest
        from scipy.stats import shapiro as shapiro
        from scipy.stats import anderson as anderson
        from statsmodels.tsa.stattools import adfuller as ad
        
        
        if isinstance(self.data, pd.DataFrame):
            n_cols = len(self.data.columns)
            col_names = self.data.columns
        elif isinstance(self.data, np.ndarray):
            n_cols = self.data.shape[1]
        
        stats = {}
        diags =  [('Mean', np.mean),
                  ('Standard Deviation', np.std),
                  ('Kurtosis', kurt), 
                  ('Skew', skew), 
                  ('Minimum', min), 
                  ('Maximum', max), 
                  ('Jarque - Bera', jb), 
                  ('Wilcoxon', wilcox), 
                  ('KS', ks),
                  ('D’Agostino and Pearson', normaltest),
                  ('Shapiro', shapiro),
                  ('Anderson', anderson),
                  ('nc', ad),
                  ('c', ad),
                  ('ct', ad),
                  ('ctt', ad)]
        
        if res:
            res_df = pd.DataFrame(self.res)
            res_df.columns = ['r' + str(i+1) for i in range(n_cols)]
            for col in range(n_cols):
                if isinstance(res_df, pd.DataFrame):
                    for i, d in enumerate(diags):
                        
                        # Calculate the statistic, adding a regression parameter for the ADF test
                        if diags[i][0] in ['nc', 'c', 'ct', 'ctt']:
                            temp = diags[i][1](res_df[res_df.columns[col]], regression = diags[i][0])
                          
                        elif diags[i][0] in ['Anderson']:
                            temp = diags[i][1](res_df[res_df.columns[col]], dist = 'norm')
                            
                        elif diags[i][0] in ['KS']:
                            temp = diags[i][1]((self.data[col_names[col]]), cdf = 'norm')
                        else:
                            temp = diags[i][1](res_df[res_df.columns[col]])
                        # Sometimes the KS test gives an error so we use a try statement
                        try:
                            #Some statistics return single floats, others return p-values and other numbers
                            # Thus we check for the type of object returned by the statistic
                            if isinstance(temp, float):
                                stats[col_names[col] + ' Residuals ' +str(diags[i][0])] = [np.round(temp, 2)]
                                    
                            else:
                                # For ADF tests we ad 'AD' in front of the name
                                if diags[i][0] in ['nc', 'c', 'ct', 'ctt']:
                                    stats[col_names[col] + ' Residuals AD_' + str(diags[i][0])] = [t for t in temp]
                                else:
                                    stats[col_names[col] + ' Residuals ' + str(diags[i][0])] = [t for t in temp]
                        except:
                            stats[col_names[col] + ' Residuals ' + str(diags[i][0])] = ['failed']
        else:
            for col in range(n_cols):
                if isinstance(self.data, pd.DataFrame):
                    for i, d in enumerate(diags):
                        
                        # Calculate the statistic, adding a regression parameter for the ADF test
                        if diags[i][0] in ['nc', 'c', 'ct', 'ctt']:
                            temp = diags[i][1]((self.data[col_names[col]]), regression = diags[i][0])
                            
                        elif diags[i][0] in ['Anderson']:
                            temp = diags[i][1]((self.data[col_names[col]]), dist = 'norm')
                            
                        elif diags[i][0] in ['KS']:
                            temp = diags[i][1]((self.data[col_names[col]]), cdf = 'norm')
                            
                        else:
                            try:
                                temp = diags[i][1]((self.data[col_names[col]]))
                            except:
                                pass
                                print('KS failed')
                        # Sometimes the KS test gives an error so we use a try statement
                        try:
                            #Some statistics return single floats, others return p-values and other numbers
                            # Thus we check for the type of object returned by the statistic
                            if isinstance(temp, float):
                                stats[col_names[col] + ' ' +str(diags[i][0])] = [np.round(temp, 2)]
                                    
                            else:
                                # For ADF tests we ad 'AD' in front of the name
                                if diags[i][0] in ['nc', 'c', 'ct', 'ctt']:
                                    stats[col_names[col] + ' AD_' + str(diags[i][0])] = [t for t in temp]
                                else:
                                    stats[col_names[col] + ' ' + str(diags[i][0])] = [t for t in temp]
                        except:
                            stats[col_names[col] + ' ' + str(diags[i][0])] = ['failed']
                            
            
   
            
                    
                elif isinstance(self.data, np.ndarray):
                    print('give a pandas dataframe')
                    #x = kurt(self.data[:,col])
                    #kurt_dict['kurt var' + str(col+1)] = x 
        
        
        if res:
            stats['Durbin-Watson on Residuals']= [self.dw[0], self.dw[1]]
            stats['Durbin-Watson on Sqaured Residuals' ] = [self.dw_sq[0], self.dw_sq[1]]
            stats['eigen_values of Φ'] = [self.eigen[0][0], self.eigen[0][1]]
            stats_df = pd.DataFrame.from_dict(stats, orient = 'index')
            stats_df.columns = ['statistic', 'p-values']  + [ 'c' + str(i) for i in range(3,7)]
            
        else:
            stats['eigen_values of Φ'] = [self.eigen[0][0], self.eigen[0][1]]
            stats_df = pd.DataFrame.from_dict(stats, orient = 'index')
            stats_df.columns = ['statistic', 'p-values']  + [ 'c' + str(i) for i in range(3,7)]
        
     
        print('Shapiro-Wilks p-value ABOVE 0.05 for Normally distributed data')
        print('KS p-value ABOVE 0.05 for Normally distributed data')
        print('JB p-value ABOVE 0.05 for Normally distributed data')
        print('D’Agostino & Pearson p-value ABOVE 0.05 for Normally distributed data')
        print('Anderson p-value ABOVE 0.05 for Normally distributed data')
        print('Shapiro p-value  ABOVE 0.05 for Normally distributed data')
        print('Wilcoxon p-value ABOVE 0.05 for Normally distributed data')
        print('ADF: pvalue is above a critical size, cannot reject unit root')
        return stats_df
        #for s in stats:
            #print("{stat}:{value}".format(stat = s, value = stats[s]))
       
    
    def acf(self, save = None, var_names = None, maxlags = 30):
        import matplotlib.pyplot as plt
        
        if var_names != None:
             mh_acf(self.data, var_names = var_names, maxlags = maxlags)
        else:
            mh_acf(self.data, maxlags = maxlags)
            
        if save != None:
            plt.savefig(self.file_path + save)
            
    def acf_sq(self, save = None, var_names = None, maxlags = 30):
        import matplotlib.pyplot as plt
        if var_names != None:
             mh_acf(self.data**2, var_names = var_names, maxlags = maxlags)
        else:
            mh_acf(self.data**2, maxlags = maxlags)
            
        if save != None:
            plt.savefig(self.file_path + save)
            
    def res_acf(self, save = None, var_names = None, maxlags = 30):
        import matplotlib.pyplot as plt
        mode = self.mode
        if var_names != None:
            mh_acf(self.res, var_names = var_names, maxlags = maxlags)
        else:
            V = ['BTC Residuals ' + mode, 'ETH Residuals ' + mode]
            mh_acf(self.res, maxlags = maxlags, var_names = V)
      
        if save != None:
            plt.savefig(self.file_path + save)
            
    def res_acf_sq(self, save = None, var_names= None, maxlags = 30):
        """
        
        Help for the res_acf_sq method in the GCov_model class
        
        
        res_acf_sq(self, save = None, var_names= None, maxlags = 30)
        """
        import matplotlib.pyplot as plt
        mode = self.mode
        if var_names != None:
            mh_acf(self.res_sq, var_names = var_names, maxlags = maxlags)
        else:
            V = ['BTC Res SQ ' + mode, 'ETH Res SQ ' + mode]
            mh_acf(self.res_sq, maxlags = maxlags, var_names = V)
        if save != None:
            plt.savefig(self.file_path + save)
    
    def res_joint(self, save = None):
        """
        
        Help for res_joint method in the GCov_model class
        
        res_joint(self, save = None)
        
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        plt.figure(figsize = (12,8), dpi = 150)
        res_df = pd.DataFrame(self.res)
        res_df.columns = ['r1', 'r2']
        sns.jointplot('r1', 'r2', data = res_df, kind = 'kde')
        if save != None:
            plt.savefig(self.file_path + save)
                
                
    
            
    def res_hist(self, save = None, var_names = None, color = None, panel = True):
        """
        Help for the res_hist method in the GCov_model class
        
        res_hist(self, save = None, var_names = None, color = None)
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        if var_names != None:
            assert isinstance(var_names, list)
            if isinstance(self.data, pd.DataFrame):
                len_v = len(self.data.columns)
            elif isinstance(self.data, np.ndarray):
                len_v = len(self.data.shape[1])
            assert len(var_names) == len_v
        if isinstance(self.data, pd.DataFrame):
            n_cols = len(self.data.columns)
        elif isinstance(self.data, np.dnarray):
            n_cols = self.data.shape[1]
        print('n_cols = ', n_cols)
        
        if var_names == None:
            if n_cols == 2:
                var_names = ['BTC', 'ETH']
            elif n_cols == 4:
                var_names = ['BTC', 'ETH', 'XRP', 'XLM']
        
        
     
        plt.figure(figsize = (12,8), dpi = 150)
        if n_cols == 2:
            plot_n = ['211', '212']
        elif n_cols == 4:
            plot_n = ['221', '222', '223', '224']
        if n_cols == 2:
                colors  = ['blue', 'black']
        elif n_cols == 4:
                colors  = ['blue', 'black', 'red', 'green']
        if panel:
            for i in range(n_cols):
            #print(i)
                plt.subplot(plot_n[i])
                plt.title('Histogram of ' + var_names[i] + ' Residuals', color = colors[i])
          
                sns.distplot(self.res[:,i], color = colors[i], rug = True)
        else:
            for i in range(n_cols):
                plt.figure()
                sns.distplot(self.res[:,i]*2, color = colors[i], rug = True) 
           
                plt.savefig("/Users/hallfam/Downloads/VAR1_ALL_res_hist_new_" + var_names[i] + ".jpg", dpi = 150)
            
        
        

       
        if save != None:
            plt.savefig(self.file_path + save)
    
    def boot(self, save = None, B = 300, diag_mode = 'mean', outlier_tol = np.inf, verbose = 0, print_iter = 1, mixed = 1, zero =1, block = 0):
        """
        
        Help for boot method in the GCov_model class
        boot(self, save = None, B = 300, diag_mode = 'mean', outlier_tol = np.inf, verbose = 0, print_iter = 1, mixed = 1, zero =1):
        """
        
        
        
        import matplotlib.pyplot as plt
        print(str(B) + ' samples')
        boot = mh_boot_residuals(self.data, outlier_tol = outlier_tol, lags = self.lags, H = self.H, n_samples = B, verbose = verbose, block = block, print_iter = print_iter, bells = 0, mixed = mixed, zero = zero, diag = 0)
        self.single_boot = boot
        VAR_boot_diagnostics(self.single_boot, n_samples = B, diag_mode = diag_mode)
        if save != None:
            plt.savefig(self.file_path + save)
    
    def means_boot(self,save = None, B = 10, diag_mode = 'mean', outlier_tol = 10, verbose = 0, print_iter = 0, OLS_boot = False, mixed = True, zero = True):
        """ Help for method means_boot in the GCov_model class
        
        means_boot(self,save = None, B = 10, diag_mode = 'mean', outlier_tol = 10, verbose = 0, print_iter = 0, OLS_boot = False, mixed = True, zero = True):
         """
        
        
        import matplotlib.pyplot as plt
      
        m = mboot(self.data, diag_mode = diag_mode, d = False, B = B, lags = self.lags, H = self.H, outlier_tol = outlier_tol,  verbose = verbose, print_iter = print_iter, OLS_boot = OLS_boot, zero = zero, mixed = mixed)
        self.list_of_boots = m
        
        VAR_boot_diagnostics(self.list_of_boots, n_samples = B)
        if save != None:
            plt.savefig(self.file_path + save)
    
    def NC_components(self, save = None, out = None, mode = None, e_first = 'e1', e_second = 'e2' , Y_first = 'Y1', Y_second = 'Y2', verbose = True, prop_size = 16):
        """
        Help for  the GCov_model class method
        
        NC_components(self, save = None, out = None, mode = None, e_first = 'e1', e_second = 'e2' , Y_first = 'Y1', Y_second = 'Y2', verbose = True)
        
        """
        if mode == None:
            mode = self.mode
        import matplotlib.pyplot as plt

        assert out != None
        from GCov import mh_acf
        if mode == None:
            mode = self.mode
        if verbose:
            print('NOTE: mode is ' + mode)
        if out == 'e_return':
            e = components(self.data,res = self.res, H = self.H, lags = self.lags, CNC_out = None, e_out ='return', mode = mode , zero = self.zero)
            return e
        elif out == 'Y_return':
            Y = components(self.data, res = self.res ,H = self.H, lags = self.lags, CNC_out = 'return', e_out = None, mode = mode , zero = self.zero)
            return Y
        elif out == 'e_joint':
            components(self.data,res = self.res, H = self.H, lags = self.lags, CNC_out = None, e_out = 'joint', mode = mode , e_first = e_first, e_second = e_second, zero = self.zero)
        elif out == 'Y_joint':
            if verbose:
                print('NOTE: if lags > 1 then the noncausal component will be in the LAST column, not the second as in the default')
            components(self.data, res = self.res, H = self.H, lags = self.lags, CNC_out = 'joint', e_out = None, mode = mode , Y_first = Y_first, Y_second = Y_second, zero = self.zero)
        elif out == 'e_plot_shared':
            components(self.data, res = self.res, H = self.H, lags = self.lags, CNC_out = None, e_out = 'e_shared', mode = mode , zero = self.zero, prop_size = prop_size )
        elif out == 'e_plot_sep':
            components(self.data, res = self.res, H = self.H, lags = self.lags, CNC_out = None, e_out = 'e_sep', mode = mode, zero = self.zero, prop_size = prop_size )
        elif out == 'e_plot':
            components(self.data, res = self.res,  H = self.H, lags = self.lags, CNC_out = None, e_out = 'e_plot', mode = mode, zero = self.zero, prop_size = prop_size )
        elif out == 'Y_plot':
            components(self.data,  res = self.res, H = self.H, lags = self.lags, CNC_out = 'CNC_shared', e_out = None , mode = mode, zero = self.zero, prop_size = prop_size )
        elif out == 'plot_all':
            components(self.data,  res = self.res, H = self.H, lags = self.lags, CNC_out = 'all', e_out = None , mode = mode, zero = self.zero, prop_size = prop_size )
        elif out == 'e_hist':
            components(self.data, res = self.res,  H = self.H, lags = self.lags, CNC_out = None, e_out = 'hist' , mode = mode, zero = self.zero, prop_size = prop_size )
        elif out == 'e_acf':
            e_return = components(self.data,  res = self.res, H = self.H, lags = self.lags, CNC_out = None, e_out = 'return' , mode = mode , zero = self.zero)
            if verbose:
                print('NOTE: there are ' + str(len(e_return.columns)) + ' elements in the decomposition of ε')
            e_return = e_return[[e_first, e_second]]
            V = [e_first + ' ' + mode, e_second + ' ' + mode]
            try:
                mh_acf(e_return.values, var_names = V )
            except:
                mh_acf(e_return, var_names = V)
        elif out == 'Y_acf':
            Y_return = components(self.data, res = self.res,  H = self.H, lags = self.lags, CNC_out = 'return', e_out = None , mode = mode, zero = self.zero )
            mh_acf(Y_return)
        if save != None:
            plt.savefig(self.file_path + save)
            
      
    def NC_irf(self, start = None, end = None, ymin = None, ymax = None, lags = None,window = 7, chol = False, save = None, title = 'BTC/ETH'):
        """
        Help for method 
        
        NC_irf(self, start = None, end = None, ymin = None, ymax = None, lags = None,window = 7)
        
        In the GCov_model class
        """
        
        if save != None:
            file_path = self.file_path
        else:
            file_path = None
        if start == None:
            start = int(len(self.data)/2) - window
            
        if end == None:
           end =  int(len(self.data)/2) + window
           
    
        lags = self.lags
        if lags > 1:
            Φ = self.block_coefs
            J = self.J
        user_matrix_data_length = len(self.data)
        assert lags > 0
        assert isinstance(lags, int)
        if lags == 1:
            # If lags == 1 then we use sympy to calculate the exact Jordan Normal Form of Φ
            # If lags > 2 we use scipy to solve an optimization problem given Φ  and J to find A
            NC_irf_external(data = self.data,
               lags = 1,
               start = start, 
               ymin =ymin,
               ymax = ymax,
               end = end, 
               tol = 100, 
               user_matrix = True,
               user_matrix_data_length = user_matrix_data_length,
               array = self.coefs[0], 
               check = False,
              mod = 'none',
              H = self.H,
              verbose = True, 
              zero = True,
              chol = chol,
              create_Z = True, graph_margin = 5, save = save, file_path = file_path, title = title)
           
        else:
            NC_irf_general(data = self.data, lags = self.lags, H = self.H, mode = self.mode, Φ = Φ, J = J, ymax = ymax, ymin = ymin, start = start, end = end, graph_margin = 5, window = window, chol = chol)
            
    def find_A(self, Φ = None, J= None, schur = True, check = False):
        """
        Help for find_A(self, Φ = None, J= None)
        
        (To be done)
        
        """
        import numpy as np
        import scipy
        
        def makeAJA(coe, check = False):
    
            import numpy as np
            import scipy as scipy 
            from sympy import Matrix
        
            
            
            U, Q = scipy.linalg.schur(coe)
            Q_inv = np.linalg.inv(Q)
            
            u = Matrix(U)
        
            P_tilde, J = u.jordan_form()
            J = np.array(J).astype(np.complex)
            P_tilde_array = np.array(P_tilde).astype(np.complex)
            P_tilde_array_inv = np.linalg.inv(P_tilde_array)
            
            if check:
                recomp_U = P_tilde_array.dot(J).dot(P_tilde_array_inv)
                result = Q.dot(recomp_U).dot(Q_inv)
                result = result.astype(np.complex).real - coe
                print('The output should be all zero, output rounded to 4 decimal places, then Q then U')
                return np.round(result, 4),np.round(Q, 4), np.round(U, 4)
            else:
                A = Q.dot(P_tilde_array)
                A_inv = np.linalg.inv(A)
                A = np.round(A.real, 8)
                J = np.round(J.real, 8)
                A_inv = np.round(A_inv.real, 8)
                return A, J, A_inv
            
        if schur:
            out = makeAJA(self.block_coefs, check = check)
            
        else:
            
            if Φ == None:
                Φ = self.block_coefs
            if J == None:
                J = self.J
            
            n = Φ.shape[0]
            
            
            def find_A_obj(theta, Φ = Φ, J = J, test = False, input_matrix = None):
                
                n = Φ.shape[0]
                if test:
                    input_matrix = np.eye(n)
                    A = input_matrix
                 
                else:
                    
                    A = np.empty(n**2)
                   
                    for i in range(n**2):
                       
                        A[i] = theta.flatten()[i]
                        
                    A = A.reshape(n,n)
             
                        
                out = Φ.dot(A) - A.dot(J)
                out = np.abs(np.trace(out.dot(out.T)))
        
        return out
        
        def A_constraint(θ, Φ = Φ, J = J):
            """
            Help for A_constraint(θ, Φ = Φ, J = J)
            
            This function sets the constraint that A_inverse Φ A = J in the optimization
            to find matrix A such that Φ = AJA_inverse
        
            paramters:
            
                θ (float): paramter to be optimized 
        
            Φ (numpy array): matrix Φ estimated by GCov
        
            J (numpy array): Jordan Matrix derived from Φ
        
            Returns:
                The objective function outputs the difference betwen AΦA_inverse and J.
                The output is the sum of the absolute value of all the parameters in the objective function
                (i.e. all the sum of the absolue value of the difference beteen the left and right sides of
                 AΦA_inverse = J)
                """
            n = Φ.shape[0]
            A_list = [θ[i] for i in range(n**2)]
            out_list  = []
            A = np.array(A_list).reshape(n,n)
            #A = np.array([
                #[θ[0], θ[1]],
                #[θ[2], θ[3]]])
            for i in range(n):
                for j in range(n):
                    out_ij = np.linalg.inv(A).dot(Φ).dot(A)[i][j] - J[i][j]
                    out_list.append(out_ij)
        
            return np.sum(np.abs(out_list))
            
            cons_A = [{"fun": A_constraint, "type": "eq"}]
                
                
                
                
                
                
            A_est = scipy.optimize.minimize(fun = find_A_obj, x0 = [1 for i in range(n**2)], method = 'BFGS', args = (Φ, J), constraints = cons_A).x
            A_est = A_est.reshape(n,n)
            A_est_inv = np.linalg.inv(A_est)
            return A_est, A_est_inv
        
def stats(data):
    import numpy as np
    from statsmodels.tsa.vector_ar.var_model import VAR as VARwrite
    from  scipy.stats import jarque_bera as jb
    from statsmodels.regression.linear_model import OLS as OLS
    from statsmodels.stats.stattools import durbin_watson as dw
    #import scikits.bootstrap as boot
    #from obspy.signal.detrend import spline as spline
    from statsmodels.tsa.stattools import adfuller as ad
    from statsmodels.tsa.arima_model import ARIMA as ARIMA
    from scipy.stats import kurtosis as kurt
    from scipy.stats import skew as skew
    from statsmodels.stats.diagnostic import kstest_normal as ks
    from scipy.stats import wilcoxon as wilcox
    from statsmodels.graphics.gofplots import qqplot as qq
    import importlib
    from scipy.stats import kurtosis as kurt
    from scipy.stats import skew as skew
    from  scipy.stats import jarque_bera as jb
    from scipy.stats import wilcoxon as wilcox
    from scipy.stats import kstest as ks
    from scipy.stats import normaltest as normaltest
    from scipy.stats import shapiro as shapiro
    from scipy.stats import anderson as anderson
    from statsmodels.tsa.stattools import adfuller as ad
    qqq = []
    L_of_stats = diags =  [('Mean', np.mean),
                  ('std', np.std),
                  ('Kurt', kurt), 
                  ('Skew', skew), 
                  ('Min', min), 
                  ('Max', max), 
                  ('JB', jb, 'p-value above 0.05 implies Normality'),
                  ('D’Agostino and Pearson', normaltest, 'p-value above 0.05 for N'),
                  ('Shapiro', shapiro,  'p-value above 0.05 for N'),
                  ('Anderson', anderson)]
    
    for i, stat in enumerate(L_of_stats):
        qqq.append((L_of_stats[i][0], L_of_stats[i][1](data)))
                    
            
    k1 =  ks(data, cdf = 'norm')[0]
    k2 = ks(data, cdf = 'norm')[1]
    K = ('ks stat and p', k1, k2)
    qqq.append(K)
    return qqq
def mod(x, y):
    """calculate modeulus"""
    import numpy as np
    return np.sqrt(x**2 + y**2)

def plot_boot(data, lines = False, save = None, title = 'Block Boot Plot', ymax = 3, color = None, select = False, col = None):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    assert isinstance(data, np.ndarray)
    n_cols = data.shape[1]
    if color == None:
        colors = ['blue', 'green', 'orange', 'red']
    plt.figure(figsize = (12,8), dpi = 150)
    if select:
        sns.distplot(data[:,col],color =  colors[col])
        plt.vlines(np.median(data[:,col]), ymin = 0, ymax = ymax)
    else:
        for i in range(n_cols):
            sns.distplot(data[:,i],color =  colors[i])
        plt.vlines(np.median(data[:,1]), ymin = 0, ymax = ymax)
        plt.vlines(np.median(data[:,2]), ymin = 0, ymax = ymax)
        plt.vlines(np.median(data[:,0]), ymin = 0, ymax = ymax)
        plt.vlines(np.median(data[:,3]), ymin = 0, ymax = ymax)
        
    if lines:
        plt.vlines(-1.3, ymin = 0, ymax = 11, color = 'orange')
        plt.vlines(0, ymin = 0, ymax = 11, color = 'green')
        plt.vlines(0.7, ymin = 0, ymax = 11, color = 'blue')
        plt.vlines(2.0, ymin = 0, ymax = 11, color = 'red')
        plt.xticks((-1.3, 0.0, 0.7, 2.0), ['-1.3', '0.0', '0.7', '2.0'])
        plt.annotate('Phi 2 = -1.3', xy=(-1.3, 3), xytext=(-0.9, 4.5),
            arrowprops=dict(facecolor='orange', shrink=0.05),
            ) 
        plt.annotate('Phi 1 = 0.7', xy=(0.8, 3), xytext=(1.2, 4),
            arrowprops=dict(facecolor='blue', shrink=0.05),
            )

        plt.annotate('Phi 3 = 0.0', xy=(0.0, 3), xytext=(0.1, 5),
            arrowprops=dict(facecolor='green', shrink=0.05),
            )

        plt.annotate('Phi 4 = 2.0', xy=(2.0, 3), xytext=(2.5, 4),
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        
    


    plt.xlabel('Parameter Values')
    plt.ylabel('Frequency')
    plt.title(title)
    if save != None:
        
        plt.savefig(save)
def get_data(dat =  None):
    """   
    if dat == 'BED':
        
        i = 0
    elif dat == 'XX':
        i = 1
    elif dat == 'ALL':
        i = 2
    else:
        i = 2
    """
    
    import pandas as pd 
    import numpy as np
    from GCov import detrend
    #from obspy.signal.detrend import spline as spline
    ETH = pd.read_csv("/Users/hallfam/Downloads/Data for Testing or CoMovements/DATA THESIS CH 1/Ethereum Historical Data - Investing.com.csv", parse_dates = ['Date'], index_col = 'Date')
    BTC = pd.read_csv("/Users/hallfam/Downloads/Data for Testing or CoMovements/Crypto including July 2019/BTC_USD Bitfinex up to July 19 2019.csv", parse_dates = ['Date'], index_col = 'Date')
    ETH = pd.DataFrame(ETH.Price)
    ETH.columns = ['ETH']
    BTC = pd.DataFrame(BTC.Price)
    BTC.columns = ['BTC']
    rmv_string(BTC, 'BTC')
    rmv_string(ETH, 'ETH')
    BTC_885 = BTC.loc['2017-01-01':'2019-06-04']
    ETH_885 = ETH.loc['2017-01-01':'2019-06-04']
    
    BED_885_raw = pd.concat([BTC_885, ETH_885], axis = 1)
    shift = 98
    BED_raw  = BED_885_raw
    BED_raw_BTC = BED_raw.loc[:,'BTC']/10
    BED_raw_ETH = BED_raw.loc[:,'ETH']
    BED_raw_new = pd.concat([BED_raw_BTC, BED_raw_ETH], axis = 1)

    BED_885_med_spline = BED_885_raw.copy()
    BED_885_med_spline = BED_885_med_spline.sub(np.median(BED_885_med_spline))
    BED_885_med_spline = BED_885_med_spline.iloc[300 + shift:550 + shift]
    BTC_885_spline = spline(BED_885_med_spline.BTC.values, order=2, dspline=29)# , plot = graph_path02) #"/Users/Frida/Desktop/Data for Testing Comovements of Exchanges/Crypto including July 2019/spline_test.jpg")

    #BED15_spline = pd.read_csv("/Users/Frida/Desktop/Data for Testing Comovements of Exchanges/Crypto including July 2019/BED5_median_DIV15.csv", parse_dates = ['Date'], index_col = 'Date')
    ETH_885_spline = spline(BED_885_med_spline.ETH.values, order=2, dspline=29) #, plot= graph_path03)
    
    BED_885_det = pd.concat([pd.Series(BTC_885_spline).div(10), pd.Series(ETH_885_spline)], axis = 1)
    BED_885_det.index = pd.date_range(start = '2018-02-03', end = '2018-10-10')
    
    
    XRP_raw = pd.read_csv("/Users/hallfam/Downloads/Data for Testing or CoMovements/XRP.csv", parse_dates = ['Date'], index_col = 'Date')
    XLM_raw = pd.read_csv("/Users/hallfam/Downloads/Data for Testing or CoMovements/XLM.csv", parse_dates = ['Date'], index_col = 'Date')
    XRP_price = XRP_raw.iloc[::-1]['Close**']
    XRP_price = XRP_price.loc['2017':]
    XLM_price = XLM_raw.iloc[::-1]['Close**']
    XLM_price = XLM_price.loc['2017':]
    XRPLM_Adj = pd.concat([XRP_price.sub(np.median(XRP_price)).div(3), XLM_price.sub(np.median(XLM_price))], axis = 1)
    XRPLM_Adj.columns = ['XRP', 'XLM']
    XRPML_window = XRPLM_Adj.loc['2018-03-25': '2018-11-29']
    XRPML_window .columns = ['XRP', 'XLM']
    XRP_det = detrend(XRPML_window.XRP, dis = 25, order = 3, plot = 0) #, plot = XRP_det_path)

    #XLM_det_path = '/Users/Frida/Desktop/Data for Testing Comovements of Exchanges/FIGURES THESIS CH 1/Figure Nov 2019/XLM_detrended.jpg'
    XLM_det = detrend(XRPML_window.XLM, dis = 25, order = 3, plot = 0) #, plot = XLM_det_path)
    XRPLM_det = pd.concat([XRP_det, XLM_det], axis = 1)

    XRPLM_det.columns = ['XRP', 'XLM']
    XRPLM_det.index = pd.date_range(start = '2018-03-25', periods = 250)
    ALL = pd.concat([BED_885_det.loc['2018-03-25':].div(1000), XRPLM_det.loc[:'2018-10-10']], axis = 1)
    
    ALL.columns = ['BTC', 'ETH', 'XRP', 'XLM']
    simVAR22m = pd.read_csv("/Users/hallfam/Downloads/Data for Testing or CoMovements/Crypto including July 2019/simVAR22m.csv")
    simVAR22m.columns = ['crap', '0', '1']
    simVAR22m = simVAR22m.loc[:, ['0', '1']]
    dats = [BED_885_det, XRPLM_det,ALL, simVAR22m, BED_raw_new, XRPML_window]
    if dat == 'BED':   
        i = 0
    elif dat == 'XX':
        i = 1
    elif dat == 'ALL':
        i = 2
    elif dat== 'SIM':
        i = 3
    elif dat == 'BED_raw':
        i = 4
    elif dat == 'XX_raw':
        i = 5
    return dats[i]



def detrend(dat, order=2, dis=100, save=None, plot=True):
    import numpy as np
    import pandas as pd
    from  GCov import mh_acf
    from obspy.signal.detrend import spline as spline
    data = dat.copy()
    data = data.sub(np.median(data))
    if not isinstance(data, np.ndarray):
        try:
            data = data.values
        except:
            raise Exception("OOPS")
    if save == None:
        new = spline(data, order=order, dspline=dis, plot = plot)
        return pd.Series(new)
    else:
        assert isinstance(save, str)
        spline(data, order=order, dspline=dis, plot = save)
        
        
def block_coefs(A):
    """
    Help for block_coefs(A)
    
    input a matrix of coefficients and this function will return a block matrix as in Davis and Song 2013
    
    paramters: 
    
    A : numpy array
    
    returns:
    
    BLOCK : numpy array block matrix with the Identity matrix on the diagonals,
            coefficient matricies along the first (block) row and zeros everywhere else
            
    """
    import numpy as np
  
    list_of_A = []
    for a in range(A.shape[0]):
        list_of_A.append(A[a])
    row1 = list_of_A
    p = A.shape[0]
    m = A.shape[1]
    if p ==1:
        BLOCK = row1
        
    elif p ==2:
        row2 = [np.eye(m), np.zeros((m,m))]
        BLOCK = np.block([row1, row2])
    elif p ==3:
        
        row2 = [np.eye(m), np.zeros((m,m)), np.zeros((m,m))]
        row3 = [np.zeros((m,m)), np.eye(m), np.zeros((m,m))]
        BLOCK = np.block([row1, row2, row3])
    elif p ==4:
        
        
        row2 = [np.eye(m), np.zeros((m,m)), np.zeros((m,m)), np.zeros((m,m))]
        row3 = [np.zeros((m,m)), np.eye(m), np.zeros((m,m)), np.zeros((m,m))]
        row4 = [np.zeros((m,m)),np.zeros((m,m)) , np.eye(m), np.zeros((m,m))]
        BLOCK = np.block([row1, row2, row3, row4])
    
    return BLOCK
    
    
    
    
    
def block_Jordan(A_block, return_eigens = False, verbose = False, n_cols = 2):
    """
    
    Help for  block_Jordan(A_block, return_eigens = False, verbose = False)
    
    Takes a symmetric block matrix as in Davis and Song (2001)
    and returns a Jordan matrix with eigen values along the main diagonal
    sorted so that the first entry is the smallest eigenvalue in absolute value
    and with ones along the subdiagonal
    
    Paramters:
    
    A_block: ndarray
    
    return_eigens: bool
        If True then a list of sorted eigenvalues is returned and nothing else
    verbose: bool
        If True the size of the input matrix is printed and the J block matrix is returned
        
    returns:
    
        a Joran matrix with eigns on the diagonal and ones along the subdiagonal
        
    """
    import numpy as np
    list_of_eigs = []
    n = A_block.shape[0] 
    
  
  
    if verbose:
        print('the size of the input matrix is ', n, ' by ', n)
        print('input matrix is', A_block)
        #Dimension of the input matrix, should be square
    for i in (range(0,n,n_cols)):
        x = np.linalg.eig(A_block[0:n_cols, i:i+n_cols])[0]
        for e in range(len(x)):
            
            try:
                list_of_eigs.append(x[e])
            except:
                list_of_eigs.append(x[e])
                
        
        #for e in range(n):
            #x = np.linalg.eig(A_block) #[0][e]
            
            #print('x is ', x)
            #list_of_eigs.append(x.real)
        
    #print('pre sorted eigs ' , list_of_eigs)
    #Sort the list by absolute value of the eigens
    list_of_eigs  = sorted(list_of_eigs, key = abs)
     
    #print('POST sorted eigs ' , list_of_eigs)
  
    if verbose:
        print('sorted list of eigenvalues is ', list_of_eigs)
    if return_eigens:
        return list_of_eigs
    else:
        B_block = np.zeros((n,n))
        
    for eig in range(n):
        if isinstance(list_of_eigs[eig], complex):
            B_block[eig][eig] = abs(list_of_eigs[eig])
        else:
            B_block[eig][eig] = list_of_eigs[eig]
        
    for ind in range(n-1):
        B_block[ind+1][ind] = 1
 
    return B_block


    
def NC_irf_general(data = None,
           lags = None,
           tol = 100, 
           start = None,
           end = None,
           user_matrix = False,
           user_matrix_data_length = 150,
           array = None, 
           check = False,
           ymax = None,
           ymin = None,
          mod = 'none',
          H = None,
          verbose = True, 
          zero = True,
          chol = False,
          create_Z = True,
                  Φ = None,
                  J = None, graph_margin = 5,
                  window = None,
                  mode = None):
    """
    Help for NC_irf_general(data = None,
           lags = None,
           tol = 100, 
           start = None,
           end = None,
           user_matrix = False,
           user_matrix_data_length = 150,
           array = None, 
           check = False,
           ymax = None,
           ymin = None,
          mod = 'none',
          H = None,
          verbose = True, 
          zero = True,
          chol = False,
          create_Z = True,
                  Φ = None,
                  J = None, graph_margin = 5,
                  window = None):
        
    Used to calculate the Noncausal Impulse Response Function for lags > 1.
    The function solves an optimization problem given Φ and J to find suitable matricies A and A_inverse
    
    Paramters:
        
    data (pandas dataframe) : data used to calculate the NC_irf
    lags (int): a positive integer, used to set the number of lags in the model
    start (int) : sets the beginning point on the graph of the NC_irf
    end (int) : sets the end point on the graph of the NC_irf
    user_matrix (bool), default = None, if given paramters are not estimated, this is used as the coefficient matrix
    use_matrix_data_length (int): a positive integer setting the length of the graph for user_matrix given
    array (numpy array) : Default None, the matrix used if user_matrix == True
    check (bool) :Default False,  used to check the working of the function
    ymax (int): maximum point on the graphs of the NC_irf
    ymin (int) : minimum point on the graphs of ht NC_irf
    mode (str): Depreciated, used to change the eigenvalues of the jordan matrix
    H (int): number of lags to use in the objective function if coeficients are being estimated
    verbose (bool): Default True, if True the function prints statements as it runs
    chol (bool) : Default False, if True the cholskey decomposition is used in the calculation of the NC_irf
    create_Z (bool): Depreciated Default False, if true then a sequence Z is created
    Φ (numpy array): Default None, if given this is used too calculate the NC_irf
    J (numpy array): Default Non, if given this is used for the Jordan matrix
    window (int): Devault None, if given this sets the width around zero for the graphs
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    if window == None:
        window = 50
    if start == None:
        start = int(len(data)/2) - window
        print('start is', start)
    if end == None:
        end = int(len(data)/2) - window
        print('start is ', start)
  
    def find_A_obj(theta, Φ, J, test = False, input_matrix = None):
        
        n = Φ.shape[0]
        if test:
            input_matrix = np.eye(n)
            A = input_matrix
         
        else:
            
            A = np.empty(n**2)
           
            for i in range(n**2):
               
                A[i] = theta.flatten()[i]
        
                                     
            A = A.reshape(n,n)
         
                
        out = Φ.dot(A) - A.dot(J)
        out = np.abs(np.trace(out.dot(out.T)))
        return out

    def A_constraint(θ, Φ = Φ, J = J):
        n = Φ.shape[0]
        A_list = [θ[i] for i in range(n**2)]
        out_list  = []
        A = np.array(A_list).reshape(n,n)
        #A = np.array([
            #[θ[0], θ[1]],
            #[θ[2], θ[3]]])
        for i in range(n):
            for j in range(n):
                out_ij = np.linalg.inv(A).dot(Φ).dot(A)[i][j] - J[i][j]
                out_list.append(out_ij)
    
        return np.sum(np.abs(out_list))
    
    cons_A = [{"fun": A_constraint, "type": "eq"}]
    
    def find_A(Φ, J, starting_values = None):
        import numpy as np
        import scipy
        n = Φ.shape[0]
        if starting_values == None:
            
            A_est = scipy.optimize.minimize(fun = find_A_obj, x0 = [1 for i in range(n**2)], method = 'BFGS', args = (Φ, J), constraints = cons_A).x
        else:
            A_est = scipy.optimize.minimize(fun = find_A_obj, x0 = starting_values, method = 'Nelder-Mead', args = (Φ, J), constraints = cons_A).x
        A_est = A_est.reshape(n,n)
        A_est_inv = np.linalg.inv(A_est)
        return A_est, A_est_inv
        
    
    
    try:
        n_cols = len(data.columns)
    except:
        n_cols = data.shape[1]
        
    def J_n1_n2(J):
        import numpy as np
        diag_less_than_one = []
       
        for i in range(J.shape[0]):
            if np.abs(J[i][i]) < 1:
                diag_less_than_one.append(J[i][i])
        n1 = len(diag_less_than_one)
        n2 = J.shape[0] - n1
        return n1, n2

    n1, n2 = J_n1_n2(J)
    
    def J1_J2(J, n1 = n1, n2 = n2):
        
        
    
        J1 = J[0:n1, 0:n1]
        if n2 != 0:
            J2 = J[n1,n1]
        else:
            print('n2 is zero')
        return J1, J2
    
    def J_powers(J, power):
        import numpy as np
        #n1, n2 = J_n1_n2(J)
        if n2 !=0:
            try:
                J_power = np.linalg.matrix_power(J, power)
            except:
                J_power = J**power
        else:
            pass
        return J_power
    
        
        return J_power

    def Fi(J, ind):
        n1, n2 = J_n1_n2(J)
        J1, J2 = J1_J2(J)
        F_i = np.zeros_like(J)
      
        
        if ind >= 0:
            F_i[0:n1,0:n1] = J_powers(J1, ind) 
             # roz stands for 'row of zeros'
            
        else:
            
            F_i[n1,n1] = -J_powers(J2, ind)
          
        return F_i
    A, A_inv = find_A(Φ, J)
    
#    if np.all( A.dot(J).dot(A_inv) != Φ):
#        print('trying 2nd time (all)')
#        A, A_inv = find_A(Φ, J, starting_values = [i for i in A.flatten()])
#    
#    if np.all(A.dot(J).dot(A_inv) != Φ):
#        print('trying 3rd time (all)')
#        A, A_inv = find_A(Φ, J, starting_values = [i for i in A.flatten()])
#    
#    if np.any(A.dot(J).dot(A_inv) != Φ):
#        print('trying 4th time (any)')
#        A, A_inv = find_A(Φ, J, starting_values = [i for i in A.flatten()])
#        
        
    print('AJA_inv = ',np.round(A.dot(J).dot(A_inv) ,2) )
    print('Φ = ',np.round(Φ,2) )
    half_length_of_df = int(len(data)/2)
    list_of_Mi = []
    if chol:
        Z = GCov_General_phi(fun = GCov_p, args = (data, lags,H, mode), VAR_lags = lags, coefs =0, resids = 1, resids_return = 1, zero = zero)
        Σ = np.cov(Z.T)
        PL = np.linalg.cholesky(Σ)
        #
    
    for ent in range(-half_length_of_df, half_length_of_df):
        
            
            
             
             
            #PL_inv = np.linalg.inv(PL)
                    
            #if lags == 2:
                #PL_inv = np.array([[PL_inv[0][0], PL_inv[0][1],0, 0 ], [PL_inv[1][0], PL_inv[1][1],0, 0 ],[0,0,0,0], [0,0,0,0]])
                  
            #elif lags == 3:
                #PL_inv = np.array([[PL_inv[0][0], PL_inv[0][1],0, 0 ,0,0], [PL_inv[1][0], PL_inv[1][1],0, 0,0 ,0 ],[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]])
                  
            
            Mi = A.dot(Fi(J, ent)).dot(A_inv)
            Mi = Mi[0:n_cols,0:n_cols]
            if chol:
                list_of_Mi.append(Mi.dot(PL))
            else:
           
                list_of_Mi.append(Mi)
            
    array_of_Mi = np.array(list_of_Mi).flatten()
    MAX = max(array_of_Mi)
    if check:
        return list_of_Mi
    else:
        list_of_M1 = []
        for i in range(len(list_of_Mi)):
            list_of_M1.append(list_of_Mi[i][0][0]/MAX)
        
        list_of_M2 = []
        for i in range(len(list_of_Mi)):
            list_of_M2.append(list_of_Mi[i][0][1]/MAX)
        
        list_of_M3 = []
        for i in range(len(list_of_Mi)):
            list_of_M3.append(list_of_Mi[i][1][0]/MAX)
        
        list_of_M4 = []
        for i in range(len(list_of_Mi)):
            list_of_M4.append(list_of_Mi[i][1][1]/MAX)
            
        M = [list_of_M1, list_of_M2, list_of_M3, list_of_M4]
        M = np.array(M)
        M = M.flatten()
   
        
        print('start is,', start)
        print('end is ', end)
        vline_loc = 0
   
        xi = np.arange(-half_length_of_df, half_length_of_df)
        xi = xi[start:end]
       
        plt.figure(figsize = (12,8), dpi = 150)
        plt.tight_layout()

        gap = end - start
        gap = int(gap/2)
  
        plt.suptitle("BTC/ETH Noncausal Impulse Response Functions (H = " + str(H) +')', fontsize=16)
         
        M1 =pd.Series(list_of_M1)
        if ymax == None:
            ymax = max(M) + int(0.5*max(M))
            print(ymax)
        if ymin == None:
            ymin = min(M) - int(0.5*max(M))
            print(ymin)
        
        
        plt.subplot(221)
        
        
        plt.title('M11')
        plt.plot(xi, M1[start:end])

        plt.vlines(vline_loc, ymin = ymin, ymax = ymax, alpha = 0.5, color = 'red')
        plt.hlines(vline_loc, xmin = -gap, xmax = gap, alpha = 1, color = 'green')
        plt.ylim(ymin = ymin, ymax = ymax )
       
        
    
        plt.subplot(222)
        M2 =pd.Series(list_of_M2)
        
        plt.title('M12')
        plt.plot(xi, M2[start:end])
       # ymin = min(M2[start:end]) - graph_margin
        #ymax = max(M2[start:end]) + graph_margin
        plt.vlines(vline_loc, ymin = ymin, ymax = ymax, alpha = 0.5, color = 'red')
        plt.hlines(vline_loc, xmin = -gap, xmax = gap, alpha = 1, color = 'green')
        plt.ylim(ymin = ymin, ymax = ymax )

    
        plt.subplot(223)
        M3 =pd.Series(list_of_M3)
        
        plt.title('M21')
        plt.plot(xi, M3[start:end])
        #ymin = min(M3[start:end]) - graph_margin
        #ymax = max(M3[start:end]) + graph_margin
        plt.vlines(vline_loc, ymin = ymin, ymax = ymax, alpha = 0.5, color = 'red')
        plt.hlines(vline_loc, xmin = -gap, xmax = gap, alpha = 1, color = 'green')
        plt.ylim(ymin = ymin, ymax = ymax )
       
    
        plt.subplot(224)
        
        M4 =pd.Series(list_of_M4)
        
        plt.title('M22')
        plt.plot(xi, M4[start:end])
        #ymin = min(M4[start:end]) - graph_margin
        #ymax = max(M4[start:end]) + graph_margin
        plt.vlines(vline_loc, ymin = ymin, ymax = ymax, alpha = 0.5, color = 'red')
        plt.hlines(vline_loc, xmin = -gap, xmax = gap, alpha = 1, color = 'green')
        plt.ylim(ymin = ymin, ymax = ymax )
        
        
        
        
def find_A_external(Φ = None, J= None, starting_values = None):
    import numpy as np
    import scipy
    """ 
   
    Help for find_A_external(Φ = None, J= None)
    Returns A and A_inverse given Φ  and J. The same as the function defined elsewhere but available direction in the
    GCov package. 
   
    Paramters:
       
    Φ (numpy array) : autoregressive coefficient matrix estimated via GCov (or some other means)
    J (numpy array) : Jordan matrix associated with Φ
    """
    
    n = Φ.shape[0]
    
    
    def find_A_obj(theta, Φ = Φ, J = J, test = False, input_matrix = None):
        """
        
        Help for fin_A_obj(theta, Φ = Φ, J = J, test = False, input_matrix = None)
        This function is used to find A given Φ and J constructed from Φ.
        
        Parameters:
            
        theta: paramters to be optimized over
        Φ (numpy array) : matrix of autoregressive coefficients estimated via GCov
        J (numpy array) : Jordan Matrix derived from Φ
        test (bool) : if true then the a matrix is used in the function
        input_matrix (numpy array): default None, if test is True then this is the matrix put in for A
            if input_matrix == None then the identity matrix is used
        """
        n = Φ.shape[0]
        if test:
            input_matrix = np.eye(n)
            A = input_matrix
         
        else:
            
            A = np.empty(n**2)
           
            for i in range(n**2):
               
                A[i] = theta.flatten()[i]
                
            A = A.reshape(n,n)
     
                
        out = Φ.dot(A) - A.dot(J)
        out = np.abs(np.trace(out.dot(out.T)))
      
        return out

    def A_constraint(θ, Φ = Φ, J = J):
        """
        
        Help for A_constraint(θ, Φ = Φ, J = J)
        
        
        This function sets the constraint that A_inverse Φ A = J in the optimization
        to find matrix A such that Φ = AJA_inverse
        
        paramters:
            
        θ (float): paramter to be optimized 
        
        Φ (numpy array): matrix Φ estimated by GCov
        
        J (numpy array): Jordan Matrix derived from Φ
        
        Returns:
        The objective function outputs the difference betwen AΦA_inverse and J.
        The output is the sum of the absolute value of all the parameters in the objective function
        (i.e. all the sum of the absolue value of the difference beteen the left and right sides of
        AΦA_inverse = J)
        """
        n = Φ.shape[0]
        A_list = [θ[i] for i in range(n**2)]
        out_list  = []
        A = np.array(A_list).reshape(n,n)
        #A = np.array([
            #[θ[0], θ[1]],
            #[θ[2], θ[3]]])
        for i in range(n):
            for j in range(n):
                out_ij = np.linalg.inv(A).dot(Φ).dot(A)[i][j] - J[i][j]
                out_list.append(out_ij)
    
        return np.sum(np.abs(out_list))
    
    cons_A = [{"fun": A_constraint, "type": "eq"}]
    if starting_values == None:
        
        A_est = scipy.optimize.minimize(fun = find_A_obj, x0 = [1 for i in range(n**2)], method = 'BFGS', args = (Φ, J), constraints = cons_A).x
    else:
         A_est = scipy.optimize.minimize(fun = find_A_obj, x0 = starting_values, method = 'BFGS', args = (Φ, J), constraints = cons_A).x
   
    A_est = A_est.reshape(n,n)
    A_est_inv = np.linalg.inv(A_est)
    return A_est, A_est_inv


def J_n1_n2_external(J):
    """ 
    
    This is the same as the function used inside another function called NC_irf_general, but this function
    is available directly from the GCov package.
    Input is a matrix (nd numpy array) with eigenvalues along the main diagonal
    from least to greatest in absolute value
    
    This is a Jordan matrix from the Jordan Canonical form of the coefficient matrix Φ
    
    Output is n1, the number of causal dimensions, and n2 the number of noncausal dimensions
    
    Parameters:
    J (ndarray) : matrix input
    
    Returns
    n1 (int) : the number of causal dimensions of the matrix Φ  which corresponds to the number of
    eigenvalues alone the main diagonal of J which are below 1 in absolute value or modulus if the eigenvalues are complx
    
    n2 (int): the number of noncausal dimensions
    """
    import numpy as np
    diag_less_than_one = []

    for i in range(J.shape[0]):
        if np.abs(J[i][i]) < 1:
            diag_less_than_one.append(J[i][i])
    n1 = len(diag_less_than_one)
    n2 = J.shape[0] - n1
    return n1, n2
 
    



###############################################################################
################## FAILED ATTEMPTS AND UNUSED FUNCTIONS #######################
###############################################################################   
    

    #def find_best_aic(data, n_tests):
#    """
#    
#    finds the 'best' ARMA model fit using aic
#    
#    Paramters:
#    
#    data: must be a series (one dimensional)
#    n_tests: how many ar and ma lags to test (max is 8 so the system does not crash)
#    """
#    import numpy as np
#    #import statsmodels.formula.api as smf
#    import statsmodels.tsa.api as smt
#    #import statsmodels.api as sm
#    #import scipy.stats as scs
#    #from arch import arch_model
#    import warnings
#    warnings.filterwarnings("ignore")
#    best_aic = np.inf
#    best_order = None
#    #best_mdl = None
#    if n_tests > 8:
#        n_tests = 8
#        print('number of tests is at most 8 or the system crashes')
#    rng = range(n_tests) 
#    for i in rng:
#        for j in rng:
#            try:
#                tmp_mdl = smt.ARMA(data.dropna(), order=(i, j)).fit(
#                    method='mle', trend='nc'
#                )
#                tmp_aic = tmp_mdl.aic
#                if tmp_aic < best_aic:
#                    best_aic = tmp_aic
#                    best_order = (i, j)
##                    best_mdl = tmp_mdl
#            except:
#                continue
#
#
#    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
#    
    
    
#def irf(data = None, lags = 1, H = 14, trend = 'nc', forecast = None, mat = None, GCov = True, shock = None):
#    import numpy as np
#    import matplotlib.pyplot as plt
#    from GCov import GCov_General_phi, GCov_p
#    
#    if GCov:
#        if lags == 1:
#            A = GCov_General_phi(fun = GCov_p, args = (data, lags,H, 1), VAR_lags = lags, coefs =1, resids = 0, acf = 0, eigen = 0, sq = 0)[0]
#            try:
#                sd1 = data[data.columns[0]].std()
#                sd2 = data[data.columns[1]].std()
#            except:
#                sd1 = np.std(data[:, 0])
#                sd2 = np.std(data[:, 1])
#            #sd1 = 1
#            #sd2 = 1
#            shock1 = np.array([sd1,0]).squeeze()
#            shock2 = np.array([0,sd2]).squeeze()  
#            
#        GCov_irf_one = []
#        GCov_irf_two = []
#        
#        for f in range(forecast):
#                
#                if shock == 'shock1':
#                    x = np.linalg.matrix_power(A, f).dot(shock1)
#                    GCov_irf_one.append(x[0])
#                    GCov_irf_two.append(x[1])
#                    
#                elif shock == 'shock2':
#                    x = np.linalg.matrix_power(A, f).dot(shock2)
#                    GCov_irf_one.append(x[0])
#                    GCov_irf_two.append(x[1])
#            
#        
#        plt.subplot(211)
#        plt.plot(GCov_irf_one, color = 'r')
#      
#        plt.title('GCov one on two, H is ' +  str(H))
#        plt.subplot(212)
#        plt.plot(GCov_irf_two, color = 'k')
#     
#        plt.title('GCov two on onem H is ' +  str(H))
#       
#        
#    else:
#        from statsmodels.tsa.api import VAR
#        
#        OLS_irf_one = []
#        OLS_irf_two = []
#      
#        if isinstance(mat, np.ndarray):
#             
#            A = mat
#        else:
#            if lags == 1:
#                A = VAR(data).fit(lags, trend = trend).coefs.squeeze()
#                shock1 = np.array([1,0]).squeeze()
#                shock2 = np.array([0,1]).squeeze()
#                
#               
#                
#            
#            for f in range(forecast):
#                
#                if shock == 'shock1':
#                    x = np.linalg.matrix_power(A, f).dot(shock1)
#                    OLS_irf_one.append(x[0])
#                    OLS_irf_two.append(x[1])
#                    
#                elif shock == 'shock2':
#                    x = np.linalg.matrix_power(A, f).dot(shock2)
#                    OLS_irf_one.append(x[0])
#                    OLS_irf_two.append(x[1])
#            
#            
#                plt.subplot(211)
#                plt.plot(OLS_irf_one, color = 'r')
#                
#                plt.title('OLS one on two')
#                plt.subplot(212)
#                plt.plot(OLS_irf_two, color = 'k')
#              
#                plt.title('OLS two on one')
#            print('shock is ' + str(shock))
def block_boot(data, 
               lags = 1, 
               VAR_trend = 'nc', 
               n_samples = 300, 
               VAR_boot = False, 
               coefs = True, 
               std_err = False,
               verbose = True, 
               zero = 1, h_range = 13, mod = 100, H = 11):

    from GCov import geo_block_sample
    import numpy as np

    SE = np.zeros((1,4))
    Z = np.zeros((1,4))

 

    if VAR_boot:
        for i in range(n_samples+1):
            if verbose:
                if i%mod == 0:
                    print(i)
            co = VAR(geo_block_sample(data)).fit(lags, trend = VAR_trend).coefs
            y1 = co.flatten()[0]
            y2 = co.flatten()[1]
            y3 = co.flatten()[2]
            y4 = co.flatten()[3]
            Y = np.array([y1,y2,y3,y4])
            Z = np.vstack((Z,Y))
        Z = Z[1:]
        list_of_p = []
        for i in range(4):
            x1 = Z[:,i]
            x1 = min(len(x1[x1 < 0])/len(x1), len(x1[x1 > 0])/len(x1))
            list_of_p.append(x1)
        return list_of_p
    
    else:
        if coefs:
            for i in range(n_samples+1):
                if verbose:
                        if i%mod == 0:
                            print(i)
                #for h in range(2,h_range):
                    
                co = GCov_model(geo_block_sample(data), lags = lags, H = H, zero = zero).coefs[0]
                #se = GCov_model(geo_block_sample(data), lags = lags, H = 11, zero = zero).se
                #try:
                    #se = se.flatten()
                #except:
                    #pass
                    
                y = co.flatten()
                Y = np.asarray(y)
                Z = np.vstack((Z,Y))
                #SE = np.vstack((SE, se))
            Z = Z[1:]
            #SE = SE[1:]
            return Z#, SE
            #list_of_p = []
            #for i in range(4):
            #    x1 = Z[:,i]
            #    x1 = min(len(x1[x1 < 0])/len(x1), len(x1[x1 > 0])/len(x1))
            #    list_of_p.append(x1)
            #return list_of_p
        elif std_err:
            for i in range(n_samples+1):
                if verbose:
                    if i%100 == 0:
                        print(i)
                co = GCov_model(geo_block_sample(data), lags = lags, H = 11, zero = zero)
                se = co.se
                try:
                    se = se.flatten()
                except:
                    pass
                SE = np.vstack((SE, se))
            return SE
        
def find_A_external2(coe,  schur = True, check = False):
        """
        Help for find_A(self, Φ = None, J= None)
        
        (To be done)
        
        """
        
        def makeAJA(coe, check = False):
    
            import numpy as np
            import scipy as scipy 
            from sympy import Matrix
        
            
            try:
                U, Q = scipy.linalg.schur(coe)
                
            except:
                U, Q = scipy.linalg.schur(coe[0])
            Q_inv = np.linalg.inv(Q)
            u = Matrix(U)
        
            P_tilde, J = u.jordan_form()
            J = np.array(J).astype(np.complex)
            P_tilde_array = np.array(P_tilde).astype(np.complex)
            P_tilde_array_inv = np.linalg.inv(P_tilde_array)
            
            if check:
                recomp_U = P_tilde_array.dot(J).dot(P_tilde_array_inv)
                result = Q.dot(recomp_U).dot(Q_inv)
                result = result.astype(np.complex).real - coe
                print('The output should be all zero, output rounded to 4 decimal places, then Q then U')
                return np.round(result, 4),np.round(Q, 4), np.round(U, 4)
            else:
                A = Q.dot(P_tilde_array)
                A_inv = np.linalg.inv(A)
                #A = np.round(A.real, 8)
                #J = np.round(J.real, 8)
                #A_inv = np.round(A_inv.real, 8)
                return A, J, A_inv
            
        if schur:
            try:
                out = makeAJA(coe, check = check)
            except:
                out = makeAJA(coe[0], check = check)
            
        else:
           pass
        return out
