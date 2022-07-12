import xml.etree.cElementTree as ET
import xml.dom.minidom
import subprocess

class FEBioModel():
    '''
        FEBio model class
    '''
    def __init__(self, name='default', encoding='ISO-8859-1', version='3.0', type='solid',
                       log_data=['ux', 'uy', 'uz'], output_data=["displacement", "stress"]):
        '''
            Initializes the FEBio model class
        
            Parameters
            ----------
            name: str, default 'default'
                Name of the output file. Must end in .feb
            encoding: str, default 'ISO-8859-1'
                XML header encoding. E.g.: 'ISO-8859-1', 'UTF-8'
            version: str, default '3.0'
                FEBio version. Assumed to be '3.0'
            type: str, default 'solid'
                Analysis type. Only 'solid' supported at the moment
            log_data: list of str
                Variables to save in the log file. See this page and adjacent ones:
                https://help.febio.org/Manuals/FEBioUser/FEBio_um_3-0-3.17.1.1.html#next
            output_data: list of str
                Variables to save in the output .xplt file.
        '''
        if version != '3.0' or type != 'solid':
            raise NotImplementedError('Only FEBio v3.0 and solid analysis supported at the moment')
        
        #Save some properties
        self.name, self.encoding= name, encoding
        self.log_data, self.output_data= log_data, output_data
        
        #Object properties
        self.node_values, self.element_values= {}, {}
        self.node_scalar_field_values, self.material_values= {}, {}
        self.step_values, self.load_curve_values = {}, {}
        self.displacement_values, self.node_set_values= {}, {}
        
        #Add high-level xml nodes
        self.root= ET.Element('febio_spec', version=version)
        ET.SubElement(self.root, "Module", type=type)
        glob= ET.SubElement(self.root, "Globals")
        cts= ET.SubElement(glob, "Constants")
        for i in ['T', 'R', 'Fc']: ET.SubElement(cts, i).text= str(0.)
        
        self.material= ET.SubElement(self.root,"Material")
        self.mesh= ET.SubElement(self.root,"Mesh")
        self.meshdomains= ET.SubElement(self.root, "MeshDomains")
        self.meshdata= ET.SubElement(self.root, "MeshData")
        #self.initialBoundary= ET.SubElement(self.root,'Boundary')
        self.step= ET.SubElement(self.root,"Step")
        self.loaddata = ET.SubElement(self.root, "LoadData")
        self.output= ET.SubElement(self.root,"Output")
        #self.initialConstraint= ET.SubElement(self.root,"Constraints")
        
        self.plotfile= ET.SubElement(self.output,"plotfile", type="febio")
        self.logfile= ET.SubElement(self.output,"logfile")
        
        #Set outputs
        for var in self.output_data: 
            ET.SubElement(self.plotfile, "var", type=var)
        ET.SubElement(self.logfile, "node_data", data=';'.join(self.log_data))
        
    def add_step(self, type='solid', name=None, **parameters):
        '''
            Add a simulation step
        '''
        name= f'Step_{len(self.step_values) + 1}' if name is None else name
        if name in self.step_values.keys():
            raise ValueError(f'Step name ({name}) cannot already exist in the model')
        
        #Overwrite default parameters
        default_parameters= {'analysis':'STATIC', 'time_steps':'10', 'step_size':'0.1', 
                             'solver':{
                                 'max_refs':'15', 'max_ups':'10', 'diverge_reform':'1', 'reform_each_time_step':'1', 
                                 'dtol':'0.001', 'etol':'0.01', 'rtol':'0', 'lstol':'0.9', 'min_residual':'1e-20',
                                 'qnmethod':'BFGS', 'rhoi':'0'
                                 },
                             'time_stepper':{
                                'dtmin':'0.01', 'dtmax':'0.1', 'max_retries':'5', 'opt_iter':'10'
                                 }
                            }
        for p,v in parameters.items(): default_parameters[p]= v
        parameters= default_parameters
        
        #Save step
        id= str(len(self.step_values) + 1)
        step= ET.SubElement(self.step, 'step', id=id, name=name)
        control= ET.SubElement(step, 'Control')
        boundary= ET.SubElement(step, 'Boundary')
        self.step_values[name]= {'type':type, 'parameters':parameters, 'id':id, 'xml_node':boundary}
        
        #Create elements
        for p,v in parameters.items():
            if isinstance(v, dict):
                param= ET.SubElement(control, p)
                for p2, v2 in v.items():
                    ET.SubElement(param, p2).text= v2
            else:
                ET.SubElement(control, p).text= v
                
        return name
    
    def add_object(self, nodes, elements, element_type, name=None):
        '''
            Add a simple one-part object mesh to the problem
            Parameters
            ----------
            nodes: array of floats
                N x D array, where N is the number of nodes and D the dimensionality (e.g. 3)
            elements: array of ints
                E x T array of ints representing node ids (0-indexed), where E is the number
                of elements and T is the number of nodes of the element type
            element_type: str
                Elment type identifier: {'tet4', 'hex8', 'penta6'}
            name: str, Optional
                Name of the object, must not exist alrady in the Model
        '''
        name= f'Object_{len(self.node_values) + 1}' if name is None else name
        if name in self.node_values.keys():
            raise ValueError(f'Object name ({name}) cannot already exist in the model')
            
        #Save nodes and elements
        self.node_values[name]= nodes
        self.element_values[name]= elements
        
        #Create nodes
        self.nodes= ET.SubElement(self.mesh, "Nodes", name=f'{name}_nodes')
        for i, n in enumerate(nodes):
            ET.SubElement(self.nodes, 'node', id=str(i+1)).text= ','.join(list(map(lambda x: f'{x:.6f}', n)))
            
        #Create elements
        self.elements= ET.SubElement(self.mesh, "Elements", name=name, type=element_type)
        for i, e in enumerate(elements):
            ET.SubElement(self.elements, 'elem', id=str(i+1)).text= ','.join(list(map(lambda x: str(x+1), e)))
            
        return name
            
    def add_element_set(self, element_ids, object_name, name=None):
        raise NotImplementedError()
            
    def add_node_set(self, node_ids, object_name, name=None):
        '''
            Add a node set
            Parameters
            ----------
            node_ids: list of ints
                List of node ids
            object_name: str
                Name of the mesh object to which the field is added. It must exist already in the Model
            name: str, Optional
                Name of the field
        '''
        name= f'NodeScalarField_{len(self.node_set_values) + 1}' if name is None else name
        if name in self.node_set_values.keys():
            raise ValueError(f'Node set name ({name}) cannot already exist in the model')
        
        if object_name not in self.node_values.keys():
            raise ValueError(f'Object name ({object_name}) must already exist in the model')
            
        #Save node set
        self.node_set_values[name]= {'nodes':node_ids, 'object':object_name}
            
        node_set= ET.SubElement(self.mesh, "NodeSet", name=name)
        for id in node_ids:
            ET.SubElement(node_set, 'n', id=str(id+1))
            
        return name
            
    def add_node_scalar_field(self, node_ids, field, object_name, name=None, node_set_name=None):
        '''
            Add a scalar nodal field
            Parameters
            ----------
            node_ids: list of ints
                List of node ids
            field: list of floats
                Value of the field for the corresponding node_id
            object_name: str
                Name of the mesh object to which the field is added. It must exist already in the Model
            name: str, Optional
                Name of the field
            node_set_name: str, Optional
                Name of existing node set corresponding with the scalar field.
                If None, or the name did not exist, a new node set is created
        '''
        name= f'NodeScalarField_{len(self.node_scalar_field_values) + 1}' if name is None else name
        if name in self.node_scalar_field_values.keys():
            raise ValueError(f'Object name ({name}) cannot already exist in the model')
        if len(node_ids) != len(field):
            raise RuntimeError(f'The number of node ids ({len(node_ids)}) is different from'
                               f'the number of field values ({len(field)})')
        if object_name not in self.node_values.keys():
            raise ValueError(f'Object name ({object_name}) must already exist in the model')
            
        #First, a corresponding node set must be created if it does not exist already
        if node_set_name is None or node_set_name not in self.node_set_values.keys():
            node_set_name= self.add_node_set(node_ids, object_name, node_set_name)   
            
        #Save scalar field
        self.node_scalar_field_values[name]= {'nodes':node_ids, 'field':field, 
                                              'object':object_name, 'nodeset':node_set_name}         
            
        #Now we can add the field
        field_node= ET.SubElement(self.meshdata, "NodeData", name=name, data_type='scalar', node_set=node_set_name)
        for id, f in enumerate(field):
            ET.SubElement(field_node, 'node', lid=str(id+1)).text= f'{f:.6f}'
            
        return name
            
    def add_material(self, type, object_name, name=None, **properties):
        '''
            Adds a material to the Model and assigns it to a mesh object
            Parameters
            ----------
            type: str
                Type of material, such as 'neo-Hookean'. Must be understood by FEBio
            object_name: str
                Name of the object mesh to which the material is added
            name: str, Optional
                Name of the material
            **properties: dict
                Material properties, such as {'density': '1.0', 'E': '1000.0','v': '0.3'}
                Must be understood by FEBio
        '''
        name= f'Material_{len(self.material_values) + 1}' if name is None else name
        if name in self.material_values.keys():
            raise ValueError(f'Material name ({name}) cannot already exist in the model')
            
        #Save material
        id= str(len(self.material_values) + 1)
        self.material_values[name]= {'type':type, 'properties':properties, 'object':object_name, 'id':id}
        
        #Add material nodes
        material= ET.SubElement(self.material, 'material', id=id, name=name, type=type)
        for p,v in properties.items():
            ET.SubElement(material, p).text= v
        ET.SubElement(self.meshdomains, 'SolidDomain', name=object_name, mat=name)
        
        return name
        
    def add_displacement(self, step_name, axis, type, scale, load_curve_name, relative=False, name=None):
        '''
            Adds a displacement prescription boundary condition (BC)
            
            Parameters
            ----------
            step_name: str
                Name of the step to which this BC applies
            axis: str
                Axis to which the BC applies: {'x', 'y', 'z'}
            type: str
                Type of prescribed displacement: {'map', }
            scale: str
                The value of the scale parameter. 
                Either a number, or a node scalar field if `type = 'map'`
            load_curve_name: str
                Name of the load curve to be assigned to this BC. 
                It must have been created beforehand
            relative: bool, default False
                Defines wheter the values are absolute or relative to the current displacement
            name: str, Optional
                Name of this displacement BC
        '''
        if step_name not in self.step_values.keys():
            raise ValueError(f'Step {step_name} must exist before adding a boundary condition to it')
        if load_curve_name not in self.load_curve_values.keys():
            raise ValueError(f'Load curve {load_curve_name} must exist before creating a boundary condition using it')
        
        name= f'Displacement_{len(self.displacement_values) + 1}' if name is None else name
        if name in self.displacement_values.keys():
            raise ValueError(f'Material name ({name}) cannot already exist in the model')
            
        #Save material
        id= str(len(self.displacement_values) + 1)
        self.displacement_values[name]= {'type':type, 'step_name':step_name, 'axis':axis, 
                                         'scale':scale, 'load_curve_name':load_curve_name}
        
        #Find nodeset associated to the field if type=map (scale is the name of the dispoacement field)
        if type == 'map':
            node_set= self.node_scalar_field_values[scale]['nodeset']
        else:
            node_set= None
            raise NotImplementedError()
        
        #Add displacement nodes
        bc= ET.SubElement(self.step_values[step_name]['xml_node'], 'bc', name=name, type='prescribe', node_set=node_set)
        ET.SubElement(bc, 'dof').text= axis
        ET.SubElement(bc, 'scale', lc=self.load_curve_values[load_curve_name]['id'], type=type).text= scale
        ET.SubElement(bc, 'relative').text= str(int(relative))
        
        return name
        
    def add_load(self):
        raise NotImplementedError()
    
    def add_load_curve(self, interpolate='SMOOTH', points=[[0,0],[1,1]], name=None):
        '''
            Add a load curve
            
            Parameters
            ----------
            interpolate: str
                Interpolation type: {'SMOOTH', 'LINEAR'}
            points: array
                P x 2 array of points where P is the number of points
            name: str, Optional
                Name of the curve
        '''
        name= f'LC_{len(self.load_curve_values) + 1}' if name is None else name
        if name in self.load_curve_values.keys():
            raise ValueError(f'Material name ({name}) cannot already exist in the model')
            
        #Save LC
        id= str(len(self.load_curve_values) + 1)
        self.load_curve_values[name]= {'type':type, 'points':points, 'id':id}
        
        #Add load controller
        load_controller= ET.SubElement(self.loaddata, "load_controller", id=id, type='loadcurve')
        ET.SubElement(load_controller, 'interpolate').text= interpolate
        points_node= ET.SubElement(load_controller, 'points')
        for p in points:
            ET.SubElement(points_node, 'point').text= f'{p[0]},{p[1]}'
            
        return name
    
    def write(self):
        '''
            Writes a nicely formatted xml .feb file
        '''
        dom = xml.dom.minidom.parseString(ET.tostring(self.root))
        xml_string = dom.toprettyxml()
        part1, part2 = xml_string.split('?>')

        with open(self.name + '.feb', 'w') as xfile:
            xfile.write(part1 + 'encoding=\"{}\"?>\n'.format(self.encoding) + part2)
            xfile.close()
                    
    def parse_output(self, steps=None):
        '''
            Parses the output log
            
            Parameters
            ----------
            steps: list of str, Optional, default None
                List of step IDs (integer) to parse from the output. Set to None to parse all
        '''
        output, lineno= {}, -1
        try:
            with open(f'{self.name}.log', 'r') as log:
                block_id= 0 #0:Outside block, 1:Inside block, 2:Inside number block
                for lineno, line in enumerate(log):
                    if line.startswith('='):
                        block_id= 1
                    elif block_id == 1:
                        if line.startswith('Step'):
                            step= int(line[line.index('=') + 1:])
                            if steps is not None and step not in steps:
                                block_id= 0
                            else:
                                output[step]= {}
                        elif line.startswith('Time'):
                            pass
                        elif line.startswith('Data'):
                            output[step]['id']= []
                            variables= line[line.index('=') + 1:].strip().split(';')
                            for var in variables: output[step][var]= []
                            block_id= 2
                        else:
                            block_id= 0
                    elif block_id == 2:
                        info= line.split(' ')
                        if len(info) - 1 == len(self.log_data):
                            output[step]['id'].append(int(info[0])-1)
                            for var, value in zip(variables, info[1:]):
                                output[step][var].append(float(value))
                        else:
                            block_id= 0
                    else:
                        block_id= 0
        except Exception as e:
            raise RuntimeError(f'Line {lineno + 1}: {e}')
        return output
        
    def run(self, **kwargs):
        #Do some checks
        if not len(self.step_values): raise ValueError('No steps defined')
        if not len(self.node_values): raise ValueError('No objects defined')
        if not len(self.material_values): raise ValueError('No materials defined')
        
        #Write .feb to disk
        self.write()
        
        #Simulate
        command= subprocess.run(['febio3', '-i', f'{self.name}.feb'])
        if command.returncode:
            raise RuntimeError(f'There was a problem with the simulation.'
                               f' Check console output or {self.name}.log')
        else:
            print('Simulation completed successfully!')
        
        #Read and return results
        return self.parse_output(**kwargs)