from timeit import default_timer

class _timer:

    def __init__(self,label='timer'):

        """
        start timer
        """

        self.label = label
        self.start_time = default_timer()


    def stop(self):

        """
        stop timer and print elapsed time
        """

        stop_time = default_timer()
        elapsed = stop_time-self.start_time
        msg = f'{self.label:24} {elapsed:6.3f} [s]\n'
        print(msg)
    

