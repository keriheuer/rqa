import numpy as np
# from scipy.stats import norm # import doesn't work
from random import gauss

class TimeSeries():

    """ Includes methods for generating time series data."""
    def __init__(self):
        pass

    def sine(self, freq, amp=1, phi=0):
        fs = 500    # (integer) sampling rate
        f = freq       # (integer) fundamental frequency of the sinusoid
        t = 1    # (integer) duration of the time series

        time = np.linspace(0, t, t * fs) # total number of samples
        signal = np.sin(2 * np.pi * f * time + phi) # for sin(B*t) the period is 2*pi/B

        return signal*amp

    def cosine(self, freq, amp=1, theta=0):
        # sample_rate = 1000
        # start_time = 0
        # end_time = 0.5
        # time = np.arange(start_time, end_time, 1/sample_rate)
        # cos_wave = amp * np.cos(2 * np.pi * freq * time + theta)
        # return cos_wave

        fs = 100    # (integer) sampling rate
        f = freq       # (integer) fundamental frequency of the sinusoid
        t = 1      # (integer) duration of the time series

        time = np.linspace(0, t, t * fs) 
        signal = np.cos(2 * np.pi * f * time) # for sin(B*t) the period is 2*pi/B
        return signal

    

    # def cosine(N, omega):
    #     """ Generates time series of sine wave.

    #     N = length of time series to generate
    #     omega = frequency of sine signal
    #     """

    #     t = np.arange(N)
    #     cosine_wave = np.cos(omega*t)
    #     plt.figure(figsize=(8,2))
    #     plt.plot(cosine_wave)
    #     plt.title('Cosine Wave')
    #     plt.xlabel('Time')
    #     plt.ylabel('Amplitude')
    #     plt.show()
    #     return cosine_wave

    # def uncorrelated_uniform_noise(self, duration=0.1, framerate=7000, show=False):
    #     """ Generates time series for uncorrelated uniform noise.
    #     """

    #     signal = UncorrelatedUniformNoise()
    #     wave = signal.make_wave(duration=400, framerate=framerate) #11025)

    #     segment = wave.segment(duration=0.1)

    #     if show == True:
    #         segment.plot(linewidth=1)
    #         decorate(xlabel='Time (s)', ylabel='Amplitude')

    #     return segment

    def white_noise(self, duration=1000, mean=0.0, std=1.0):
        """
        Generates time series for uncorrelated uniform (Gaussian) noise.

        :param duration: integer, length of time series
        :param mean: float, average value of noise
        :param std: float, standard deviation of the noise
        """

        white_noise = np.array([gauss(mean, std) for i in range(duration)])
        return white_noise
    
 
    # methods to generate Brownian motion
    #     REF: https://towardsdatascience.com/brownian-motion-with-python-9083ebc46ff0
    def gen_random_walk(self, x0=0, n_step=100):
        """
        Generate motion by random walk

        Arguments:
            n_step: Number of steps

        Returns:
            A NumPy array with `n_steps` points
        """
        # Warning about the small number of steps
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")

        w = np.ones(n_step)*x0

        for i in range(1,n_step):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1,-1])
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))

        return w

    def gen_normal(self,x0=0, n_step=100):
        """
        Generate motion by drawing from the Normal distribution

        Arguments:
            n_step: Number of steps

        Returns:
            A NumPy array with `n_steps` points
        """
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")

        w = np.ones(n_step)*x0

        for i in range(1,n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))

        return w

    def disrupted_brownian(self, N, M):
        """ Generates time series for disrupted Brownian motion.
        N = number of iterations to return
        M = number of points/random variates to use
        """

        r = norm.rvs(size=M)
        x = np.empty((M))

        for i in np.arange(M-1):
            x[i+1] = x[i] + (2*r[i])

        disrupted = x[1:N]

        return disrupted

    def rossler(self, a,b,c,t,tf,h):

        def derivative(r,t):
            x = r[0]
            y = r[1]
            z = r[2]
            return np.array([- y - z, x + a * y, b + z * (x - c)])

        time = np.array([]) #Empty time array to fill for the x-axis
        x = np.array([]) #Empty array for x values
        y = np.array([]) #Empty array for y values
        z = np.array([]) #Empty array for z values
        r = np.array([1.0, 1.0, 1.0]) #Initial conditions array

        while (t <= tf ):
            time = np.append(time, t)
            z = np.append(z, r[2])
            y = np.append(y, r[1])
            x = np.append(x, r[0])

            k1 = h*derivative(r,t)
            k2 = h*derivative(r+k1/2,t+h/2)
            k3 = h*derivative(r+k2/2,t+h/2)
            k4 = h*derivative(r+k3,t+h)
            r += (k1+2*k2+2*k3+k4)/6
            t = t + h

        # fig, ax = plt.subplots(figsize=(12,3), ncols=3)
        # fig.suptitle('Phase Space Portrait')
        # ax[0].plot(x, y)
        # ax[1].plot(y, z)
        # ax[2].plot(x, z)
        # for axis, xlabel, ylabel in zip(ax.ravel().tolist(), ['x', 'y', 'x'], ['y', 'z', 'z']):
        #   axis.set_xlabel(r'$%s$' % xlabel)
        #   axis.set_ylabel(r'$%s$' % ylabel)
        # plt.subplots_adjust(wspace=0.25)
        # plt.show()

        return time, x,y,z

    def logistic_map(self, r,x,drift,N,M):
        """ Generates time series for logistic map.
            REF: https://www.r-bloggers.com/logistic-map-feigenbaum-diagram/

            r = bifurcation parameter
            x = initial value
            N = number of iterations
            M = number of iteration points to returns
        """

        z = np.empty((N))
        z[0] = x
        for i in np.arange(N-1):
            z[i+1] = r* z[i] * (1-z[i])

        lm = z[N-M:N]

        # add linear trend if given
        lm = lm+ drift*np.arange((len(lm)))

        # plt.figure(figsize=(8,2))
        # plt.title('Logistic Map')
        # plt.plot(lm)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.show()
        return lm

    # def superimpose(self, N, signal1_type, freq1, signal2_type, freq2):

    #   """ Generates time series of superimposed waves.

    #   N = length of time series to generate
    #   omega = frequency of sine signal
    #   """
    #   t = np.arange(N)

    #   if signal1_type == 'Sine':
    #     ts1 = np.sin(freq1*t)
    #   else:
    #     ts1 = np.cos(freq1*t)
    #   if signal2_type == 'Sine':
    #     ts2 = np.sin(freq2*t)
    #   else:
    #     ts2 = np.cos(freq2*t)

    #   mixed = ts1 + ts2

    #   plt.figure(figsize=(8,2))
    #   plt.plot(t, mixed)
    #   plt.title('%s + %s' % (signal1_type, signal2_type))
    #   plt.xlabel('Time')
    #   plt.ylabel('Amplitude')
    #   plt.show()

    #   return mixed

    def henon(self, length=10000, x0=None, a=1.4, b=0.3, discard=500):
        """Generate time series using the Henon map.

        REF: https://github.com/manu-mannattil/nolitsa/blob/master/nolitsa/data.py

        Parameters
        ----------
        length : int, optional (default = 10000)
            Length of the time series to be generated.
        x0 : array, optional (default = random)
            Initial condition for the map.
        a : float, optional (default = 1.4)
            Constant a in the Henon map.
        b : float, optional (default = 0.3)
            Constant b in the Henon map.
        discard : int, optional (default = 500)
            Number of steps to discard in order to eliminate transients.

        Returns
        -------
        x : ndarray, shape (length, 2)
            Array containing points in phase space.
        """
        x = np.empty((length + discard, 2))

        if not x0:
            x[0] = (0.0, 0.9) + 0.01 * (-1 + 2 * np.random.random(2))
        else:
            x[0] = x0

        for i in range(1, length + discard):
            x[i] = (1 - a * x[i - 1][0] ** 2 + b * x[i - 1][1], x[i - 1][0])

        return x[discard:]

    def normalize_ts(self):
        '''
        Normalize a time series to have zero mean and unit standard deviation.
        Inputs:
            self.ts : input time series (numpy array or list)
        Outputs:
            z : normalized time series (numpy array)
        '''
        # if type(self.ts) is list:
        #     self.ts = np.array(self.ts)
        # if type(self.ts) is not np.ndarray:
        #     print('input time series must by a numpy array or list. exiting.')
        #     return 

        # normalize time series 
        z = (self.ts - np.mean(self.ts)) / np.std(self.ts)

        return z