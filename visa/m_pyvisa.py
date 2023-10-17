import pyvisa


# ---------------------------------------------------------------------------------------------------

class c_pyvisa:

    # ----------------------------------------------------------------------------------------------

    def __init__(self,debug=False):
        
        """
        interface to pyvisa package
        debug: whether or not to log pyvisa debugging info to screen
        """

        if debug:
            pyvisa.log_to_screen()

        self.resource_manager = pyvisa.ResourceManager()
        
        # holds resources
        self.resources = {}

    # ----------------------------------------------------------------------------------------------

    def _open_resource(self,resource_id):

        """
        try to open a resource
        """

        try:
            resource = self.resource_manager.open_resource(resource_id)
        except Exception as ex:
            print('\n*** ERROR ***\ncouldnt connect to resource\n')
            print(ex,'\n')
            return None

        print('\n*** resource info ***')
        print(resource.query('*IDN?'))

        return resource

    # ----------------------------------------------------------------------------------------------

    def print_available_resources(self,try_to_connect=False):

        """
        loop over all available resources, try to connect, and print info if successful
        """

        resource_ids = self.resource_manager.list_resources()
        print('\navailable resources:\n'+'\n'.join(resource_ids),'\n')

        if try_to_connect:

            for resource_id in resource_ids:   

                print('\nResource:',resource_id)

                _instr = self._open_resource(resource_id)
                if _instr is None:
                    continue
                else:
                    _instr.close()

    # ----------------------------------------------------------------------------------------------

    def open_resource(self,resource_id,resource_handle='_'):

        """
        open a resource and add it to the resource dict
        """

        if resource_handle in self.resources:
            msg = f'\n*** ERROR ***\na resource with the handle:\n \"{resource_handle}\"\n'
            msg += 'already exists! pick a different handle or close open resource\n'
            print(msg)
            return

        print('\n*** opening resource ***')
        print('resource:',resource_id)
        print('handle:',resource_handle)

        _instr = self._open_resource(resource_id)
        self.resources[resource_handle] = _instr

    # ----------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    pv = c_pyvisa()

    pv.print_available_resources(try_to_connect=True)
    pv.open_resource('ASRL33::INSTR')
















