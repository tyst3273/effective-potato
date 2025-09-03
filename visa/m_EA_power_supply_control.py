
import pyvisa as pv
import os
import threading
import time

# --------------------------------------------------------------------------------------------------

def get_power_supply_control():

    """
    just a macro to instantiate class
    """

    ps = c_EA_power_supply_control()
    ps.connect_resource(resource='TCPIP0::192.168.0.2::5025::SOCKET')
    ps.print_id()
    return ps

# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------

class c_EA_power_supply_control:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,visa_lib=None,interval=0.01):

        r"""
        connect to resource manager

        if visa_lib=None, try to autodetect.

        for windows, visa_lib=r'C:\Windows\System32\visa64.dll' (probably)
        """

        self.interval = interval # seconds
        self.logging = False
        self.log_thread = None

        if visa_lib is None:
            self.resoure_manager = pv.ResourceManager()
        else:
            self.resoure_manager = pv.ResourceManager(visa_lib)

    # ----------------------------------------------------------------------------------------------

    def print_resources(self):

        """
        try to list all available resources and thier ids
        """

        print('\nresource manager:\n  '+str(self.resoure_manager))

        _resources = self.resoure_manager.list_resources()
        msg = '\nresources:\n  '+'\n  '.join(_resources)
        print(msg)

        for _r in _resources:

            print('\n'+_r)

            try:
                _r = self.resoure_manager.open_resource(_r)
                print(_r.query('*IDN?'))

            except Exception as ex:
                print(ex)
                continue

# ----------------------------------------------------------------------------------------------

    def connect_resource(self,resource='ASRL4::INSTR'):

        """
        connect to resource
        """

        self.resource = resource

        try:
            self.instr = self.resoure_manager.open_resource(resource)

            self.instr.timeout = 5000
            self.instr.read_termination = '\n'
            self.instr.write_termination = '\n'

            _id = self.instr.query('*IDN?')

        except Exception as ex:
            print(f'\nfailed to connect/get id for {resource}\n  '+str(ex)+'\n')

        # clear errors
        while True:

            err = self.instr.query('SYST:ERR?')
            if err.split(',')[0] == '0':
                break   

    # ----------------------------------------------------------------------------------------------

    def print_id(self):

        """
        print id for resource
        """

        print('\nresource:',self.resource)
        print(self.instr.query('*IDN?'))
        # print('version:',self.instr.query("SYST:VERS?"))

    # ----------------------------------------------------------------------------------------------

    def remote(self):

        """
        switch power supply to remote (usb) control
        """

        # if ps_control.instr.query('SYST:LOCK:OWNER?').strip() == 'NONE':
        #     self.instr.write('SYST:LOCK ON')
        self.instr.write('SYST:LOCK ON')

    # ----------------------------------------------------------------------------------------------

    def local(self):

        """
        switch power supply to local (knob) control
        """

        msg = '\nWARNING! dont use local()\n'
        print(msg)

        # if ps_control.instr.query('SYST:LOCK:OWNER?').strip() == 'REMOTE':
        #     self.instr.write('SYST:LOCK OFF')
        self.instr.write('SYST:LOCK OFF')

    # ----------------------------------------------------------------------------------------------

    def set_voltage(self,voltage):

        """
        set power supply setpoint voltage (volts)
        """

        self.voltage_setpoint = voltage
        return self.instr.write(f'VOLT {voltage}')

    # ----------------------------------------------------------------------------------------------

    def set_current(self,current):

        """
        set power supply setpoint current (amps)
        """

        self.current_setpoint = current
        return self.instr.write(f'CURR {current}')

    # ----------------------------------------------------------------------------------------------

    def output_on(self):

        """
        switch on the output
        """

        return self.instr.write('OUTP ON')

    # ----------------------------------------------------------------------------------------------

    def output_off(self):

        """
        switch off the output
        """

        return self.instr.write('OUTP OFF')

    # ----------------------------------------------------------------------------------------------

    def measure_voltage(self):

        """
        measure output voltage (volts)
        """

        return float(self.instr.query('MEAS:VOLT?').strip().strip('V'))

    # ----------------------------------------------------------------------------------------------
    
    def measure_current(self):

        """
        measure output current (amps)
        """

        return float(self.instr.query('MEAS:CURR?').strip().strip('A'))

    # ----------------------------------------------------------------------------------------------

    def start_logging(self,filename):

        """
        start the logger in a background thread
        """

        # if already logging, do nothing
        if self.logging:
            return

        # start the logger
        self.logging = True
        self.log_thread = threading.Thread(
            target=self._logger, args=(filename, self.interval), daemon=True)
        self.log_thread.start()

    # ----------------------------------------------------------------------------------------------

    def stop_logging(self):

        """
        go and kill and join the logger
        """

        self.logging = False

        if self.log_thread:
            self.log_thread.join()
            self.log_thread = None

    # ----------------------------------------------------------------------------------------------

    def _logger(self, filename, interval, fmt=' 012.6f'):

        """
        a task to run in background to read data and write to file
        """

        with open(filename,'w') as f:

            f.write('   TIME[S]      V-OUT[V]     I-OUT[A]     V-SET[V]     I-SET[A]\n')
            t0 = None

            while self.logging:

                try:

                    _V = self.measure_voltage()
                    _I = self.measure_current()

                    _V0 = self.voltage_setpoint
                    _I0 = self.current_setpoint

                    if t0 is None:
                        _t = 0.0
                        t0 = time.time()

                    else:
                        _t = time.time()-t0

                    _line = f'{_t:{fmt}} {_V:{fmt}} {_I:{fmt}} {_V0:{fmt}} {_I0:{fmt}}\n'
                    f.write(_line)
                    f.flush()

                except Exception as ex:
                    print('logging error:',ex)
                    print(self.instr.query('SYST:ERR?'))

                time.sleep(interval)

    # ----------------------------------------------------------------------------------------------

    def _backup_files(self,filename):

        """
        create a backup of files in dir with prefix = filename
        """

        _ls = os.listdir()
        _rename = []
        _inds = []

        for _f in _ls:
            if _f.startswith(filename) and _f.endswith('bup'):
                _rename.append(_f)
                _ind = int(_f.split('.')[-2])
                _inds.append(_ind)

        _inds = sorted(_inds,reverse=True)
        _count = len(_inds)
        for ii, _ind in enumerate(_inds):
            _old = filename + f'.{_ind}.bup'
            _new = filename + f'.{_count-ii}.bup'
            os.rename(_old,_new)

        os.rename(filename,filename+'.0.bup')

    # ----------------------------------------------------------------------------------------------

    def start_sequencing(self,voltage,current,filename='ps.log'):

        """
        start sequencing at constant current/voltage and write results to file
        """

        # if os.path.exists(filename):
        #     self._backup_files(filename)

        print(f'\nsequencing to file {filename}')
        print(f'voltage: {voltage} [V]')
        print(f'current: {current} [A]')

        # need to be usb enabled
        self.remote()

        # set the setpoints
        self.set_voltage(voltage)
        self.set_current(current)

        # switch on the powersupply
        self.output_on()

        # start logging data
        self.start_logging(filename)

    # ----------------------------------------------------------------------------------------------

    def stop_sequencing(self):

        """
        stop sequencing, close file, etc
        """

        # join logging thread
        self.stop_logging()

        # shutoff power supply
        self.output_off()

        # switch back to local mode - breaks shit somehow ?
        # self.local()

    # ----------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    ps = c_EA_power_supply_control() #visa_lib='@py')
    # ps.print_resources()
    ps.connect_resource(resource='TCPIP0::192.168.0.2::5025::SOCKET')
    ps.print_id()

    # ps = get_power_supply_control()

    ps.start_sequencing(15,0.001)
    time.sleep(10)

    ps.set_voltage(10)
    time.sleep(10)

    ps.set_voltage(11)
    time.sleep(10)

    ps.set_current(4)
    time.sleep(10)
  
    ps.stop_sequencing()