def sine(N, omega):
    """ Generates time series of sine wave.

    N = length of time series to generate
    omega = frequency of sine signal
    """

    t = np.arange(N)
    sine_wave = np.sin(omega*t)
    plt.figure(figsize=(8,2))
    plt.plot(sine_wave)
    plt.title('Sine Wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    return sine_wave

def cosine(N, omega):
    """ Generates time series of sine wave.

    N = length of time series to generate
    omega = frequency of sine signal
    """

    t = np.arange(N)
    cosine_wave = np.cos(omega*t)
    plt.figure(figsize=(8,2))
    plt.plot(cosine_wave)
    plt.title('Cosine Wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    return cosine_wave

def uncorrelated_uniform_noise(duration=0.1, framerate=7000, show=False):
    """ Generates time series for uncorrelated uniform noise.
    """

    signal = UncorrelatedUniformNoise()
    wave = signal.make_wave(duration=400, framerate=framerate) #11025)

    segment = wave.segment(duration=0.1)

    if show == True:
        segment.plot(linewidth=1)
        decorate(xlabel='Time (s)', ylabel='Amplitude')

    return segment

def white_noise(N, mean=0, std=0.1, factor=750):
    """ Generates time series for white noise.

        N = number of samples to return
        factor = amplication factor for plotting purposes
    """

    """
    Generates a sine wave

    :param x: x value, scalar or vector between 0 and ..
    :param sigma_noise: standard deviation of the noise added to the data
    """

    mean = mean
    std = std
    timestep = 0.002 # 500 Hz
    endtime = 45 #sec
    numsam = int(endtime/timestep)
    t = np.linspace(0, endtime, numsam)
    samples = np.random.normal(mean, std, size=numsam)
    data = np.zeros((numsam,2))

    ## Decide Amplification factor
    data[:, 0] = [i*factor for i in samples]

    white_noise.ts = data[:N, 0]

    plt.figure(figsize=(8,2))
    plt.plot(white_noise.ts)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('White Noise')
    plt.show()

    return white_noise.ts

def sine(N, omega):
    """ Generates time series of sine wave.

    N = length of time series to generate
    omega = frequency of sine signal
    """

    t = np.arange(N)
    sine_wave = np.sin(omega*t)
    plt.figure(figsize=(8,2))
    plt.plot(sine_wave)
    plt.title('Sine Wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    return sine_wave

def cosine(N, omega):
    """ Generates time series of cosine wave.

    N = length of time series to generate
    omega = frequency of cosine signal
    """

    t = np.arange(N)
    cosine_wave = np.cos(omega*t)
    plt.figure(figsize=(8,2))
    plt.plot(cosine_wave)
    plt.title('Cosine Wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    return cosine_wave


class Brownian():
    """
    A Brownian motion class constructor.

    REF: https://towardsdatascience.com/brownian-motion-with-python-9083ebc46ff0
    """
    def __init__(self,x0=0):
        """
        Init class
        """
        assert (type(x0)==float or type(x0)==int or x0 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0)
    
    def gen_random_walk(self,n_step=100):
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
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1,-1])
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
    def gen_normal(self,n_step=100):
        """
        Generate motion by drawing from the Normal distribution
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w

def disrupted_brownian(N, M):
    """ Generates time series for disrupted Brownian motion.
    N = number of iterations to return
    M = number of points/random variates to use
    """

    r = norm.rvs(size=M)
    x = np.empty((M))

    for i in np.arange(M-1):
        x[i+1] = x[i] + (2*r[i])

    disrupted = x[1:N]

    plt.figure(figsize=(8,2))
    plt.title('Damped Random Walk')
    plt.plot(disrupted)
    plt.xlabel('Time (arbitrary units)')
    plt.ylabel('Amplitude')
    plt.show()

    return disrupted

def rossler(a,b,c,t,tf,h):

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

    fig, ax = plt.subplots(figsize=(12,3), ncols=3)
    fig.suptitle('Phase Space Portrait')
    ax[0].plot(x, y)
    ax[1].plot(y, z)
    ax[2].plot(x, z)
    for axis, xlabel, ylabel in zip(ax.ravel().tolist(), ['x', 'y', 'x'], ['y', 'z', 'z']):
      axis.set_xlabel(r'$%s$' % xlabel)
      axis.set_ylabel(r'$%s$' % ylabel)
    plt.subplots_adjust(wspace=0.25)
    plt.show()

    return time, x,y,z

def logistic_map(r,x,drift,N,M):
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

    plt.figure(figsize=(8,2))
    plt.title('Logistic Map')
    plt.plot(lm)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    return lm

def superimpose(N, signal1_type, freq1, signal2_type, freq2):

  """ Generates time series of superimposed waves.

  N = length of time series to generate
  omega = frequency of sine signal
  """
  t = np.arange(N)

  if signal1_type == 'Sine':
    ts1 = np.sin(freq1*t)
  else:
    ts1 = np.cos(freq1*t)
  if signal2_type == 'Sine':
    ts2 = np.sin(freq2*t)
  else:
    ts2 = np.cos(freq2*t)

  mixed = ts1 + ts2

  plt.figure(figsize=(8,2))
  plt.plot(t, mixed)
  plt.title('%s + %s' % (signal1_type, signal2_type))
  plt.xlabel('Time')
  plt.ylabel('Amplitude')
  plt.show()

  return mixed

def henon(length=10000, x0=None, a=1.4, b=0.3, discard=500):
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

white_noise = np.load("white_noise_ts.npy")
sine = np.load("sine_ts.npy")
super_sine = np.load("super_sine_ts.npy")
logi = np.load("logistic_map_ts.npy")
dis = np.load("brownian_ts.npy")

white_noise_rp = RecurrencePlot(white_noise, metric='euclidean', silence_level=2, normalize=True, recurrence_rate=0.05, tau=2, dim=2)
sine_rp = RecurrencePlot(sine, metric='euclidean', silence_level=2, normalize=True, recurrence_rate=0.05, tau=18, dim=6)
super_sine_rp = RecurrencePlot(super_sine, metric='euclidean', silence_level=2, normalize=True, recurrence_rate=0.05,tau=8, dim=5)
logi_rp = RecurrencePlot(logi, metric='euclidean', silence_level=2, normalize=True, recurrence_rate=0.05, tau=5, dim=10)
dis_rp = RecurrencePlot(dis, metric='euclidean', silence_level=2, normalize=True, recurrence_rate=0.05, tau=2, dim=2)

signals = [white_noise_rp, sine_rp, super_sine_rp, logi_rp, dis_rp]