using DSP, Plots

const mesh_size = 51
const st_max = 0.8
const cr_max = 0.7
const incr_max = 0.8
const frames = 1000

const scharr = [+047+47im  +0+162im  -047+047im
                +162+00im  +0+000im  -162+000im
                +047-47im  +0-162im  -047-047im] 


gaussian_distribution = ℯ.^(-(range(-ℯ, ℯ, 1*2+1).^2))
const diffusion_real = gaussian_distribution/sum(gaussian_distribution)
const diffusion_imag = complex.(diffusion_real)
const diffusion1 = 2
gaussian_distribution = ℯ.^(-(range(-ℯ, ℯ, diffusion1*2+1).^2))
const diffusion_kernel1 = complex.(gaussian_distribution/sum(gaussian_distribution))
const diffusion2 = 8
gaussian_distribution = ℯ.^(-(range(-ℯ, ℯ, diffusion2*2+1).^2))
const diffusion_kernel2 = complex.(gaussian_distribution/sum(gaussian_distribution))
const diffusion3 = 32
gaussian_distribution = ℯ.^(-(range(-ℯ, ℯ, diffusion3*2+1).^2))
const diffusion_kernel3 = complex.(gaussian_distribution/sum(gaussian_distribution))
const diffusion4 = 128
gaussian_distribution = ℯ.^(-(range(-ℯ, ℯ, diffusion4*2+1).^2))
const diffusion_kernel4 = complex.(gaussian_distribution/sum(gaussian_distribution))

rho = rand(Float64, (mesh_size, mesh_size))*0.01 .+ 1
p = zeros(ComplexF64, mesh_size, mesh_size)

for i=1:mesh_size
    for j=1:mesh_size
        p[i, j] = ((j-mesh_size/2)im - (i-mesh_size/2))*100
    end
end

for i=1:7
    rho[rand(1:mesh_size), rand(1:mesh_size)] += rand(1:100)
end

mesh2 = zeros(ComplexF64, mesh_size+2, mesh_size+2)
mesh3 = zeros(ComplexF64, mesh_size+2, mesh_size+2)
mesh4 = zeros(ComplexF64, mesh_size+2, mesh_size+2)


@gif for epoch=0:frames
    global rho, p, mesh2, mesh3, mesh4

    coeff = (epoch/(epoch+frames/10))

    incr = incr_max * coeff
    st = st_max * coeff
    cr = cr_max * coeff

    if epoch%31 == 0
        rho = conv(diffusion_real, diffusion_real, rho)[2:end-1, 2:end-1]
        p   = conv(diffusion_imag, diffusion_imag, p)[2:end-1, 2:end-1]
    end

    pressure = conv(diffusion_imag, diffusion_imag, complex.(rho))
    mesh0 = conv(diffusion_kernel1, diffusion_kernel1, complex.(rho))[diffusion1:end-diffusion1+1, diffusion1:end-diffusion1+1]
    mesh1 = conv(scharr, mesh0)[2:end-1, 2:end-1]

    if epoch%3 == 0
        mesh2 = conv(diffusion_kernel2, diffusion_kernel2, mesh1)[diffusion2+1:end-diffusion2, diffusion2+1:end-diffusion2]
    end
    if epoch%11 == 0
        mesh3 = conv(diffusion_kernel3, diffusion_kernel3, mesh2)[diffusion3+1:end-diffusion3, diffusion3+1:end-diffusion3]
    end
    if epoch%29 == 0
        mesh4 = conv(diffusion_kernel4, diffusion_kernel4, mesh3)[diffusion4+1:end-diffusion4, diffusion4+1:end-diffusion4]
    end
    

    force = 16*mesh1 .+ 8*mesh2 .+ 4*mesh3 .+ 2*mesh4 .- pressure

    rho .+= 0.00001

    vel = p ./ rho
    acc = force[2:end-1, 2:end-1] ./ rho
    vel .= vel .+ acc*incr
    p .= vel .* rho

    d_p = zeros(ComplexF64, mesh_size+2, mesh_size+2)
    d_rho = zeros(Float64, mesh_size+2, mesh_size+2)

    for y=1:mesh_size
        for x=1:mesh_size
            vlc = vel[y, x]
            arg = angle(vlc)
            mag = abs(vlc)
            mass = rho[y, x] * (mag/(mag+1))
            x += 1
            y += 1
            if -1.00π < arg < -0.75π
                mass_w  = mass * 4*(-arg-0.75π)/π * st
                mass_sw = mass * 4*(+arg+1.00π)/π * cr
                d_rho[y, x] -= mass_w + mass_sw
                d_rho[y  , x-1] += mass_w
                d_rho[y-1, x-1] += mass_sw
                d_p[y, x] -= (mass_w + mass_sw) * vlc
                d_p[y  , x-1] += mass_w  * vlc
                d_p[y-1, x-1] += mass_sw * vlc
            elseif -0.75π < arg < -0.50π
                mass_sw = mass * 4*(-arg-0.50π)/π * cr
                mass_s  = mass * 4*(+arg+0.75π)/π * st
                d_rho[y, x] -= mass_sw + mass_s
                d_rho[y-1, x-1] += mass_sw
                d_rho[y-1, x  ] += mass_s
                d_p[y, x] -= (mass_sw + mass_s) * vlc
                d_p[y-1, x-1] += mass_sw * vlc
                d_p[y-1, x  ] += mass_s  * vlc
            elseif -0.50π < arg < -0.25π
                mass_s  = mass * 4*(-arg-0.25π)/π * st
                mass_se = mass * 4*(+arg+0.50π)/π * cr
                d_rho[y, x] -= mass_s + mass_se
                d_rho[y-1, x  ] += mass_s
                d_rho[y-1, x+1] += mass_se
                d_p[y, x] -= (mass_s + mass_se) * vlc
                d_p[y-1, x  ] += mass_s  * vlc
                d_p[y-1, x+1] += mass_se * vlc
            elseif -0.25π < arg < 0.00π
                mass_se = mass * 4*(-arg-0.00π)/π * cr
                mass_e  = mass * 4*(+arg+0.25π)/π * st
                d_rho[y, x] -= mass_se + mass_e
                d_rho[y-1, x+1] += mass_se
                d_rho[y  , x+1] += mass_e
                d_p[y, x] -= (mass_se + mass_e) * vlc
                d_p[y-1, x+1] += mass_se * vlc
                d_p[y  , x+1] += mass_e  * vlc
            elseif 0.00π < arg < 0.25π
                mass_e  = mass * 4*(-arg+0.25π)/π * st
                mass_ne = mass * 4*(+arg-0.00π)/π * cr
                d_rho[y, x] -= mass_e + mass_ne
                d_rho[y  , x+1] += mass_e
                d_rho[y+1, x+1] += mass_ne
                d_p[y, x] -= (mass_e + mass_ne) * vlc
                d_p[y  , x+1] += mass_e  * vlc
                d_p[y+1, x+1] += mass_ne * vlc
            elseif 0.25π < arg < 0.50π
                mass_ne = mass * 4*(-arg+0.50π)/π * cr
                mass_n  = mass * 4*(+arg-0.25π)/π * st
                d_rho[y, x] -= mass_ne + mass_n
                d_rho[y+1, x+1] += mass_ne
                d_rho[y+1, x  ] += mass_n
                d_p[y, x] -= (mass_ne + mass_n) * vlc
                d_p[y+1, x+1] += mass_ne * vlc
                d_p[y+1, x  ] += mass_n  * vlc
            elseif 0.50π < arg < 0.75π
                mass_n  = mass * 4*(-arg+0.75π)/π * st
                mass_nw = mass * 4*(+arg-0.50π)/π * cr
                d_rho[y, x] -= mass_n + mass_nw
                d_rho[y+1, x  ] += mass_n
                d_rho[y+1, x-1] += mass_nw
                d_p[y, x] -= (mass_n + mass_nw) * vlc
                d_p[y+1, x  ] += mass_n  * vlc
                d_p[y+1, x-1] += mass_nw * vlc
            elseif 0.75π < arg < 1.00π
                mass_nw = mass * 4*(-arg+1.00π)/π * cr
                mass_w  = mass * 4*(+arg-0.75π)/π * st
                d_rho[y, x] -= mass_nw + mass_w
                d_rho[y+1, x-1] += mass_nw
                d_rho[y  , x-1] += mass_w
                d_p[y, x] -= (mass_nw + mass_w) * vlc
                d_p[y+1, x-1] += mass_nw * vlc
                d_p[y  , x-1] += mass_w  * vlc
            end
            y -= 1
        end
    end
    p .= p .+ d_p[2:end-1, 2:end-1]
    rho .= rho .+ d_rho[2:end-1, 2:end-1]

    println(epoch)

    heatmap(rho, size=(880, 800), clim=(0, 10))
end

