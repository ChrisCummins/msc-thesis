/* integrate_eom_kernel.cl */
__kernel void integrate_eom(
    __global float4* oldPosition,
    __global float4* oldVelocity,
    __global float4* newPosition,
    __global float4* newVelocity,
    __local float4* localPos,
    int numElectrons,
    float deltaTime
                            )
{
    unsigned int tid = get_local_id(0);
    unsigned int gid = get_global_id(0);
    unsigned int localSize = get_local_size(0);

    // Number of tiles we need to iterate
    // all numTiles must be multiples of localsize to ensure an integer division without reminder
    unsigned int numTilesElectrons = numElectrons / localSize;

    // pick work-item for which we will calculate the new position
    float4 pos = oldPosition[gid];
    float4 vel = oldVelocity[gid];

    float m=1.0;

    // initilize acceleration to 0
    float2 acc     = (float2)(0.0,0.0);
    float2 acc_aux = (float2)(0.0,0.0);

    // printf("tid=%d gid=%d pos=(%+e,%+e,%+e,%+e)\n",tid,gid,pos.x,pos.y,pos.z,pos.w);

    // first calculate the electron-electron repulsion
    for(int i=0;i<numTilesElectrons;++i)
    {
        // load one tile into local memory
        int idx = i * localSize + tid;
        localPos[tid] = oldPosition[idx];

        // synchronize to make sure data is available for processing
        barrier(CLK_LOCAL_MEM_FENCE);

        // calculate mutual acceleration
        acc_aux.x=0.0;
        acc_aux.y=0.0;
        for(int j = 0; j < localSize; ++j)
        {
            // calculate acceleration caused by particle j on particle i
            float4 r = localPos[j] - pos;
            float distSqr = r.x * r.x  +  r.y * r.y;

            distSqr += 1.0e-9;
            // Pauli force repulsion
            float s=m*m*localPos[j].w*pos.w/distSqr;
            {
                acc_aux.x +=-s*r.x;
                acc_aux.y +=-s*r.y;
            }
        }
        // Synchronize so that next tile can be loaded
        acc.x+=acc_aux.x;
        acc.y+=acc_aux.y;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // second calculate the forces exerted by the positive background charges
    {
        acc.x+=-0.5*m*pos.x;
        acc.y+=-0.5*m*pos.y;
    }
    // simplistic integration
    {
        vel.x += acc.x * deltaTime;
        vel.y += acc.y * deltaTime;

        pos.x += vel.x * deltaTime;
        pos.y += vel.y * deltaTime;
    }

    newPosition[gid] = pos;
    newVelocity[gid] = vel;
}
