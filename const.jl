Tgas = 40
T = Tgas

abundc = 1.e-4
abundo = 3.e-4
abundhe = 0.1
abundS = 0.0 #0.6e-6
abundSi = 3.37e-6
abar = 1+abundhe*4.0+abundc*12.0+abundo*16.0+abundSi*28.0+abundS*32.0

mp = 1.673e-24 #in gramm
rho = 1.e-24   #g/cm^3
nhtot = 100.0  #rho/(abar*mp)

H = nhtot
O = abundo*H
Ck = abundc*H
E = Ck
H2 = 1.e-40
Hk = 1.e-40

idx_H = 1
idx_E = 2
idx_Hj = 3
idx_H2 = 4
idx_Cj = 5
idx_O = 6
idx_CO = 7
idx_Tgas = 8
idx_C = 9
idx_Oj = 10
idx_Si = 11
idx_Sij = 12

user_deff = 1.0
user_dust_to_gas_ratio = 1.0
user_Ghab = 1.0 #1.e1*1.69/2.0
user_crate = 5.e-17
user_Av = 100  #1e-6
user_H2self = 1.95e-9 #2.5e-7
user_COself = 1.0 #0.5
user_Tdust = 40.0
user_uv_ion = 0.0


function T_interval(T, Tmin, Tmax)
    return 1.0*(Tmin < T < Tmax)
end

fshield_dust = exp(-2.5e0*user_Av)
G_dust = user_Ghab*fshield_dust
function h_gr(E, Tgas)
    ch34 = 5.087e2*Tgas^1.586e-2
    ch35 = -0.4723e0-1.102e-5*log(Tgas)

    if E == 0
        phi = 1e20
    else
        phi = G_dust*sqrt(T)/E
    end

    if phi < 1e-6
        user_h_gr = 1.225e-13*user_dust_to_gas_ratio
    else
        phi = 1.225e-13*user_dust_to_gas_ratio/(1e0+(8.074e-6*phi^1.378e0)*(1e0+ch34*phi^ch35))
    end

    return user_h_gr
end

AV_conversion_factor = 6.289e-22
function user_xr_ion(E)

    N_Htot = user_Av/(user_dust_to_gas_ratio * AV_conversion_factor)
    p1 = log10(N_Htot/1e18)

    if p1 < 0e0
      p1 = 0e0
    elseif p1 > 5e0
      p1 = 5e0
    end

    if E/nhtot < 1e-4
        p2 = -4.0
    elseif E/nhtot > 0.1
        p2 = -1.0
    else
        p2 = log10(E/nhtot)
    end

    f4 = 1.06+4.08e-2*p2+6.51e-3*p2^2e0
    f5 = 1.90+0.678*p2+0.113*p2^2e0
    f6 = 0.990-2.74e-3*p2+1.13e-3*p2^2e0

    ion = f4*(-15.6 - 1.10 * p1 + 9.13e-2 * p1^2e0) + f5 * 0.87 * exp(-((p1 - 0.41) / 0.84)^2e0)
    user_xr_ion = 1e1^(ion)

    return user_xr_ion
end

gamma_chx = 2.94e-10*user_Ghab*exp(-2.5*user_Av)
function beta(O)
    return 5e-10*O/(5e-10*O+gamma_chx)
end

function H_nuclei(H, H2, Hk)
    return H + H2*2 + Hk
end


Te = Tgas*8.617343e-5
invT = 1/Tgas
lnTe = log(Te)

function ncrinv(H, H2, Hk, Tgas)
    t4log = log10(Tgas)-4e0
    ch5 = 1e0/(1e1^(3e0-0.416e0*t4log-0.327e0*t4log*t4log))
    ch6 = 1e0/(1e1^(4.845e0-1.3e0*t4log+1.62e0*t4log*t4log))
    return 2e0*H2/H_nuclei(H, H2, Hk)*(ch6-ch5)+ch5
end
function h2var0(H, H2, Hk, Tgas)
    return 1e0/(1e0+H_nuclei(H, H2, Hk)*ncrinv(H, H2, Hk, Tgas))
end

function ch3(Tgas)
    h2_low_n = 1.2e-16*sqrt(T)*exp(-(1e0+5.48e0/Te))/(sqrt(4.5e3)*exp(-(1e0+5.48e0/(8.617343e-5*4.5e3))))
    h2_high_n = max(1.1e-9*T^0.135e0*exp(-5.2e4*invT),1e-40)
    return max(h2_low_n/h2_high_n, 1e-100)
end
h2var1 = ch3(Tgas)^h2var0(H, H2, Hk, Tgas)
function ch4(Tgas)
    h2_high_n_h2 = max(6.5e-8*exp(-5.2e4*invT)*invT^0.485e0,1e-40)
    h2_low_n_h2 = 5.996e-30*T^4.1881e0*exp(-54657.4e0*invT)/(1e0+6.761e-06*T)^5.6881e0
    return max(h2_low_n_h2/h2_high_n_h2,1e-100)
end
h2var2 = ch4(Tgas)^h2var0(H, H2, Hk, Tgas)
fA = 1e0/(1e0+1e4*exp(-6e2/(user_Tdust+1e-40)))
