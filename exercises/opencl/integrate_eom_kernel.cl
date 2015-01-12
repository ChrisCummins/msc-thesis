/*
 * integrate_eom_kernel.cl - Molecular dynamics simulation.
 *
 * See:
 *
 *   http://quantumdynamics.wordpress.com/2012/07/15/computational-physics-gp-programming-interacting-many-body-simulation-with-opencl/
 */

__kernel void integrate_eom(
    __global float4* oldPosition,
    __global float4* oldVelocity,
    __global float4* newPosition,
    __global float4* newVelocity,
    __local float4* localPos,
    int numElectrons,
    float deltaTime) {
    // Global ID (used to index particles).
    unsigned int gid = get_global_id(0);
    // Local ID (used to cooperate within work group).
    unsigned int tid = get_local_id(0);
    // Local size (num. of threads executing kernel).
    unsigned int localSize = get_local_size(0);

    // Number of tiles. All numTiles must be multiples of localsize to
    // ensure an integer division without reminder.
    unsigned int numTilesElectrons = numElectrons / localSize;

    // Pick work-item for which we will calculate the new position.
    float4 pos = oldPosition[gid];
    float4 vel = oldVelocity[gid];

    float m = 1.0;

    // initilize acceleration to 0.
    float2 acc = (float2)(0.0, 0.0);
    float2 acc_aux = (float2)(0.0, 0.0);

    // Calculate the electron-electron repulsion.
    for (int i = 0; i < numTilesElectrons; i++) {
        // Load one tile into local memory.
        localPos[tid] = oldPosition[i * localSize + tid];

        // Wait for all other work items in the work-group to do the
        // same.
        barrier(CLK_LOCAL_MEM_FENCE);

        // Calculate mutual acceleration.
        acc_aux.x = 0.0;
        acc_aux.y = 0.0;
        for (int j = 0; j < localSize; j++) {
            // Calculate acceleration caused by particle j on particle i.
            float4 r = localPos[j] - pos;
            float distSqr = r.x * r.x  +  r.y * r.y;

            distSqr += 1.0e-9;
            // Pauli force repulsion.
            float s = m * m * localPos[j].w * pos.w / distSqr;
            {
                acc_aux.x += - s * r.x;
                acc_aux.y += - s * r.y;
            }
        }

        acc.x+=acc_aux.x;
        acc.y+=acc_aux.y;

        // Wait for all work items in the work-group to finish before
        // moving on to the next block.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Calculate the forces exerted by the positive background
    // charges.
    {
        acc.x += -0.5 * m * pos.x;
        acc.y += -0.5 * m * pos.y;
    }

    // Simplistic integration.
    {
        vel.x += acc.x * deltaTime;
        vel.y += acc.y * deltaTime;

        pos.x += vel.x * deltaTime;
        pos.y += vel.y * deltaTime;
    }

    // Update global memory.
    newPosition[gid] = pos;
    newVelocity[gid] = vel;
}
