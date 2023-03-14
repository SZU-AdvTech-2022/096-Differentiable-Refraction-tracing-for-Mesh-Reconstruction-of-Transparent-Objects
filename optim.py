# coding=gbk
import os
import torch
import time
import captured_data
import numpy as np
import DiffRender as Render
import config

import pymeshlab

Float = captured_data.Float
device= captured_data.device

class Meshlabserver:
    def __init__(self):
        self.meshlab_remesh_srcipt = """
        <!DOCTYPE FilterScript>
        <FilterScript>
         <filter name="Remeshing: Isotropic Explicit Remeshing">
          <Param value="3" isxmlparam="0" name="Iterations" type="RichInt" description="Iterations" tooltip="Number of iterations of the remeshing operations to repeat on the mesh."/>
          <Param value="false" isxmlparam="0" name="Adaptive" type="RichBool" description="Adaptive remeshing" tooltip="Toggles adaptive isotropic remeshing."/>
          <Param value="false" isxmlparam="0" name="SelectedOnly" type="RichBool" description="Remesh only selected faces" tooltip="If checked the remeshing operations will be applied only to the selected faces."/>
          <Param value="{}" isxmlparam="0" name="TargetLen" type="RichAbsPerc" description="Target Length" min="0" max="214.384" tooltip="Sets the target length for the remeshed mesh edges."/>
          <Param value="180" isxmlparam="0" name="FeatureDeg" type="RichFloat" description="Crease Angle" tooltip="Minimum angle between faces of the original to consider the shared edge as a feature to be preserved."/>
          <Param value="true" isxmlparam="0" name="CheckSurfDist" type="RichBool" description="Check Surface Distance" tooltip="If toggled each local operation must deviate from original mesh by [Max. surface distance]"/>
          <Param value="1" isxmlparam="0" name="MaxSurfDist" type="RichAbsPerc" description="Max. Surface Distance" min="0" max="214.384" tooltip="Maximal surface deviation allowed for each local operation"/>
          <Param value="true" isxmlparam="0" name="SplitFlag" type="RichBool" description="Refine Step" tooltip="If checked the remeshing operations will include a refine step."/>
          <Param value="true" isxmlparam="0" name="CollapseFlag" type="RichBool" description="Collapse Step" tooltip="If checked the remeshing operations will include a collapse step."/>
          <Param value="true" isxmlparam="0" name="SwapFlag" type="RichBool" description="Edge-Swap Step" tooltip="If checked the remeshing operations will include a edge-swap step, aimed at improving the vertex valence of the resulting mesh."/>
          <Param value="true" isxmlparam="0" name="SmoothFlag" type="RichBool" description="Smooth Step" tooltip="If checked the remeshing operations will include a smoothing step, aimed at relaxing the vertex positions in a Laplacian sense."/>
          <Param value="true" isxmlparam="0" name="ReprojectFlag" type="RichBool" description="Reproject Step" tooltip="If checked the remeshing operations will include a step to reproject the mesh vertices on the original surface."/>
         </filter>
        </FilterScript>
        """
        pid = str(os.getpid())
        path = config.tmp_path
        self.ply_path = f"{path}/temp_{pid}.ply"
        self.remeshply_path = f"{path}/remesh_{pid}.ply"
        self.script_path = f"{path}/script_{pid}.mlx"
        self.cmd = config.meshlabserver_cmd + \
            ' -i ' + self.ply_path + \
            ' -o ' + self.remeshply_path + \
            ' -s ' + self.script_path
        self.cmd = self.cmd + " 1>/dev/null 2>&1"

        self.ms=pymeshlab.MeshSet()
        print(self.cmd)
        
    def remesh(self, scene, remesh_len):
        meshlab_remesh_srcipt = self.meshlab_remesh_srcipt.format(remesh_len)
        with open(self.script_path, 'w') as script_file:
            script_file.write(meshlab_remesh_srcipt)
        scene.mesh.export(self.ply_path)
        pass
        self.ms.load_new_mesh(self.ply_path)
        self.ms.load_filter_script(self.script_path)
        self.ms.apply_filter_script()
        self.ms.save_current_mesh(self.remeshply_path)


        scene.update_mesh(self.remeshply_path) 




class Loss_calculator:
    def __init__(self, scene, data, HyperParams):
        self.scene = scene
        self.data = data
        self.HyperParams = HyperParams
        self.ray_view  = data.ray_view_generator()
        self.silh_view = data.silh_view_generator()

    def vh_loss(self):
        scene = self.scene
        data = self.data
        #����������ʧ
        vh_loss = 0
        for v in np.arange(0,72,9):
            # index =  (V_index+v)%72
            index =  next(self.silh_view)
            screen_pixel, valid, mask, origin, ray_dir, camera_M = data.get_view(index)
            silhouette_edge = scene.silhouette_edge(origin[0])
            index, output = scene.primary_visibility(silhouette_edge, camera_M, origin[0], detach_depth=True) #����Ҫ�õ������������������ཻ����Ϣ
            vh_loss += (mask.view((data.resy,data.resx))[index[:,1],index[:,0]] - output).abs().sum()

        return vh_loss

    def sm_loss(self):
        scene = self.scene
        #����ƽ����ʧ
        dihedral_angle = scene.dihedral_angle() # cosine of angle [-1,1]
        #dihedral_angle = -torch.log(1+dihedral_angle)
        dihedral_angle = torch.log2(2 - dihedral_angle)
        sm_loss = dihedral_angle.sum()

        return sm_loss

    def ray_loss(self):
        scene = self.scene
        data = self.data
        #���������ʧ
        V_index = next(self.ray_view)
        target, valid, mask, origin, ray_dir, camera_M = data.get_view(V_index)
        render_out_ori, render_out_dir, render_mask = scene.render_transparent(origin, ray_dir) #����õ�������������֮���������ߵ���Ϣ

        screen_pixel = target
        target = screen_pixel  - render_out_ori.detach()
        target = target/target.norm(dim=1, keepdim=True)

        diff = (render_out_dir - target)
        valid_mask = valid * render_mask[:,0]
        ray_loss = (diff[valid_mask]).pow(2).sum()

        return ray_loss    

    def all_loss(self):
        scene = self.scene
        data = self.data
        HyperParams = self.HyperParams

        zeroloss = torch.tensor(0, device=device)
        ray_loss = self.ray_loss()\
            if HyperParams['ray_w'] !=0 else zeroloss
        vh_loss = self.vh_loss()\
            if HyperParams['vh_w'] !=0 else zeroloss
        sm_loss = self.sm_loss()\
            if HyperParams['sm_w'] !=0 else zeroloss


        LOSS = HyperParams['ray_w'] * 217.5 /data.resy/data.resy * ray_loss\
            + HyperParams['vh_w'] * 217.5 / data.resy * vh_loss\
            + HyperParams['sm_w'] * scene.mean_len/10 * sm_loss
        return LOSS, f'ray={ray_loss:g} vh={vh_loss:g} sm={sm_loss:g}'

def get_data(HyperParams):
    Redmi_cam = ['tiger','pig','horse','rabbit']
    Pointgray_cam = ['hand', 'mouse', 'dog', 'monkey']
    name = HyperParams['name']

    if name in Pointgray_cam:
        data = captured_data.Data_Pointgray(HyperParams)
    elif name in Redmi_cam:
        data = captured_data.Data_Redmi(HyperParams)
    else: 
        assert False
    return data

def optimize(HyperParams, output=True):

    def interp_L(start, end, it, Pass):
        assert it <= Pass-1
        step = (end - start)/(Pass-1)
        return it*step + start

    def interp_R(start, end, it, Pass):
        return 1/interp_L(1/start, 1/end, it, Pass)

    def limit_hook(grad):
        max = 1
        if torch.isnan(grad).any():
            print("nan in grad")
        grad[torch.isnan(grad)] = 0
        grad[grad>max]=max
        grad[grad<-max]=-max
        return grad

    def setup_opt(scene, lr, HyperParams):

        init_vertices = scene.vertices
        parameter = torch.zeros(init_vertices.shape, dtype=Float, requires_grad=True, device=device)     
        parameter.register_hook(limit_hook)
        opt = torch.optim.SGD([parameter], lr=lr, momentum = HyperParams['momentum'] , nesterov =True) #�Ż�������

        return init_vertices, parameter, opt


    name = HyperParams['name']
    scene = Render.Scene(f"{config.data_path}{name}_vh.ply")
    meshlabserver = Meshlabserver()
    data = get_data(HyperParams)
    Render.intIOR = HyperParams['IOR']
    Render.resy = data.resy
    Render.resx = data.resx
    Render.device = device
    Render.Float = Float

    loss_calculator = Loss_calculator(scene, data, HyperParams)


    start_time = time.time()

    for i_pass in range(HyperParams['Pass']):
        remesh_len = interp_R(HyperParams['start_len'], HyperParams['end_len'], i_pass, HyperParams['Pass'])
        lr = interp_R(HyperParams['start_lr'], HyperParams['lr_decay'] * HyperParams['start_lr'], i_pass, HyperParams['Pass'])

        print(f'remesh_len {remesh_len:g} lr {lr:g}')
        meshlabserver.remesh(scene, remesh_len)
        init_vertices, parameter, opt = setup_opt(scene, lr, HyperParams) #��ʼ������Ͳ���

        for it in range(HyperParams['Iters']):
            # Zero out gradients before each iteration
            opt.zero_grad()

            vertices = init_vertices + parameter  #�Բ������ݶȣ����¶���λ�ã��Ӷ�����trimesh�е�����������ģ��
            scene.update_verticex(vertices)
            if torch.isnan(vertices).any():
                print("nan in vertices")

            loss, loss_str = loss_calculator.all_loss()
            if torch.isnan(loss).any():
                print("nan in LOSS")
            loss.backward()

            if it%100==0 and output:
            #if it and output:
                print(f'Iteration {it}: {loss_str} maxgrad={parameter.grad.abs().max():g}')

            opt.step()

    print(f"optimize time : {time.time() - start_time}")

    return scene

if __name__ == "__main__":

    HyperParams = config.HyperParams
    scene = optimize(HyperParams)
    os.makedirs(config.result_path, exist_ok=True)
    _ = scene.mesh.export(f"{config.result_path}{HyperParams['name']}_recons.ply")
