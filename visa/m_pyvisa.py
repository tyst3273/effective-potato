import pyvisa
import tomllib
import os


# --------------------------------------------------------------------------------------------------

def warn(warning_msg=None,exception=None):

    """
    print a warning but dont crash
    """

    msg = '\n*** WARNING ***\n'
    if warning_msg is not None:
        msg += warning_msg+'\n'
    if exception is not None:
        msg += '\nexception:\n'+str(exception)+'\n'
    print(msg)

# --------------------------------------------------------------------------------------------------

def crash(err_msg=None,exception=None):

    """
    stop execution in a safe way
    """

    msg = '\n*** ERROR ***\n'
    if err_msg is not None:
        msg += err_msg+'\n'
    if exception is not None:
        msg += '\nException:\n'+str(exception)+'\n'
    print(msg)
    raise KeyboardInterrupt

# --------------------------------------------------------------------------------------------------

def check_file(file_name):

    """
    check if the specified file exists
    """

    if not os.path.exists(file_name):
        msg = f'the file:\n  \'{file_name}\' \nwas not found!'
        crash(msg)

# --------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------

class c_pyvisa:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,debug=False,config_file='config.toml'):
        
        """
        interface to pyvisa package
        debug: whether or not to log pyvisa debugging info to screen
        """

        # set a bunch of defaults
        self.resource_id = 'ASRL33::INSTR'
        self.resource_handle = 'ea_ps'
        self.visa_library_path = None
        self.write_termination = None
        self.read_termination = None
        self.baud_rate = None
        self.log_to_screen = False

        # parse config file
        self._parse_config_file(config_file)

        # set up debugging
        if debug or self.log_to_screen:
            pyvisa.log_to_screen()

        if self.visa_library_path is None:
            self.resource_manager = pyvisa.ResourceManager()
        else:
            self.resource_manager = pyvisa.ResourceManager(self.visa_library_path)
        
        # holds resources
        self.resources = {}

    # ----------------------------------------------------------------------------------------------

    def _parse_config_file(self,config_file):

        """
        parse .ini file for config options
        """

        allowed_keys = ['resource_id','resource_handle','visa_library_path','write_termination',
                    'read_termination','baud_rate','log_to_screen']

        check_file(config_file)

        with open(config_file,'rb') as f:
            config = tomllib.load(f)
        
        # loop over all the keys in the config file and set them
        for key in config.keys():
            if key not in allowed_keys:
                msg = f'the config option \'{key}\' is uknown.\nignorning it!'
                warn(msg)
            setattr(self,key,config[key])

    # ----------------------------------------------------------------------------------------------

    def _open_resource(self,resource_id):

        """
        try to open a resource
        """

        print('\nattempting to connect to resource:',resource_id)
        try:
            resource = self.resource_manager.open_resource(resource_id)
            print('sucess!')
        except Exception as ex:
            warn('couldnt connect to resource',ex)
            return None

        print('\nattempting to query resource:',resource_id)
        try:
            info = resource.query('*IDN?')
            print('sucess!')
            print('\n*** resource info ***\n',info)
        except Exception as ex:
            msg = 'couldnt query the resource\n. maybe the termination characters are wrong?'
            warn(msg,ex)

        return resource

    # ----------------------------------------------------------------------------------------------

    def print_available_resources(self,try_to_connect=False):

        """
        loop over all available resources, try to connect, and print info if successful
        """

        resource_ids = self.resource_manager.list_resources()
        print('\navailable resources:\n - '+'\n - '.join(resource_ids),'\n')

        if try_to_connect:

            for resource_id in resource_ids:   
                
                print('\n------------------------------------------------------------------\n')

                _instr = self._open_resource(resource_id)
                if _instr is None:
                    continue
                else:
                    _instr.close()

    # ----------------------------------------------------------------------------------------------

    def open_resource(self,resource_id=None,resource_handle=None):

        """
        open a resource and add it to the resource dict
        """

        if resource_id is None:
            resource_id = self.resource_id
        if resource_handle is None:
            resource_handle = self.resource_handle

        if resource_handle in self.resources:
            msg = f'a resource with the handle:\n \"{resource_handle}\"\n'
            msg += 'already exists! pick a different handle or close open resource\n'
            warn(msg)
            return

        print('\n*** opening resource ***')
        print('resource:',resource_id)
        print('handle:',resource_handle)

        _instr = self._open_resource(resource_id)

        if self.write_termination is not None:
            _instr.write_termination = self.write_termination
        if self.read_termination is not None:
            _instr.read_termination = self.read_termination
        if self.baud_rate is not None:
            _instr.baud_rate = self.baud_rate

        self.resources[resource_handle] = _instr

    # ----------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    pv = c_pyvisa()

    pv.print_available_resources(try_to_connect=True)
    pv.open_resource('ASRL33::INSTR')
















