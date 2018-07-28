# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 02:15:15 2018

@author: kennedy
"""

class Regressioon():  
    def __init__(self, dataframe, feature, start_date, end_date):
        
        '''
        Arguments:
            dataframe: dataframe we are trying to forecast from the web.DataReader
            dataname: This is the required name of the company deta we want to perform
                        regression
            feature: It indicates the dependent variable we would be prediction
            start_date: As implied signifies the start date of the stock we intend to predict
            end_date:   As implied signifies the end date of the stock we intend to predict
            
        '''
        
        self.dataframe = dataframe
        self.feature = feature
        self.start_date = start_date
        self.end_date = end_date
        
        #Load data using datetime for the
        #start and end date
        #start_date = datetime(1976, 1, 1)
        #end_date = datetime(2018, 7, 16)
        try:
            if self.start_date == None and self.end_date == None:
                raise('No data to plot regression')
            else:
                self.regress()
        except ValueError:
            raise
        finally:
            pass
        
    def regress(self):
        
        #get data
        self.dataframe = web.DataReader(dataframe, "yahoo", start_date, end_date)
        
        #define the feature vector we would be using for 
        #to plot our regression
        df = dataframe[[feature]]
        
        #Its a dataframe so we have to convert
        #it into a numerical data
        #We also dont need this since the data is already in float
        #df.info() to check datatype
        #df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
        
        df['Volatility'] = df[self.feature] - df[self.feature].shift(1)
        df = df.dropna()
        
        #linear regression model
        from sklearn.linear_model import LinearRegression
        
        #this we would be using to draw our regression line
        Xf1 = np.arange(1, len(df)+ 1)
        Xf2 = (x1**2).astype(np.uint64)
        Xf3 = (x1**3).astype(np.uint64)
        Xf4 = (x1**4).astype(np.uint64)
        
        #put our numpy array in a list
        Xf = [Xf1, Xf2, Xf3, Xf4]
        #transpose and reshape our data into (Nx4)Dimensions
        Xf = np.reshape(Xf, (4, len(df))).T
        
        
        #create a regression class
        regress = LinearRegression(n_jobs = -1)
        regress.fit(Xf, df[feature])
        
        #get the coefficients and intercept
        coeffs = regress.coef_
        intercept = regress.intercept_
        
        #create a Regression and residual column
        #in out dataframe
        df['Regression'] = intercept + coeffs[0] * x1 + coeffs[1] * x2 + coeffs[2] * x3 + coeffs[3] * x4 
        df['Residuals'] = df[self.feature] - df['Regression']
        df['Upper regresss bound'] = df['Regression'] + df['Regression'].std()
        df['Lower regresss bound'] = df['Regression'] - df['Regression'].std()
        
        #plot 2D chart of result
        df[[self.feature, 'Regression', 'Upper regresss bound', 'Lower regresss bound',]].plot(title = 'Polynomial Regression line for {}'.format(self.dataname))
        plt.legend(loc = 'best')
        plt.grid(True)
        plt.show()


c = Regressioon(dataframe = 'IBM', feature = 'Close', start_date = datetime(1976, 1, 1), end_date = datetime(2018, 7, 16))
c.regress()