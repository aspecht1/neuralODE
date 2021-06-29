@register user_xr_ion(E)
@register T_interval(Tgas, Tmin, Tmax)
@register h2var0(H, H2, Hk, Tgas)
@register H_nuclei(H, H2, Hk)
@register beta(O)
@register ncrinv(H, H2, Hk, Tgas)
@register ch3(Tgas)
@register ch4(Tgas)
@register h_gr(E, Tgas)
small_network = @reaction_network begin

    # Rates for the NL97 network implemented as CHEMISTRYNETWORK 5 in FLASH
    ###### reactions listed for H+ formation/destruction
    exp(-32.71396786e0+13.5365560e0*log(Tgas*8.617343e-5)-5.73932875e0*(log(Tgas*8.617343e-5)^2)+1.56315498e0*(log(Tgas*8.617343e-5)^3)-0.28770560e0*(log(Tgas*8.617343e-5)^4)+3.48255977e-2*(log(Tgas*8.617343e-5)^5)-2.63197617e-3*(log(Tgas*8.617343e-5)^6)+1.11954395e-4*(log(Tgas*8.617343e-5)^7)-2.03914985e-6*(log(Tgas*8.617343e-5)^8)), H + E --> Hk + E + E  #1
    2.753e-14*(315614e0*1/Tgas)^1.500e0/((1e0+(115188e0*1/Tgas)^0.407e0)^2.242e0), Hk + E --> H  #2
    h_gr(E, Tgas)*H_nuclei(H, H2, Hk)/max(E,1e-40), Hk + E --> H  #3
    user_crate, H --> Hk + E  #4
    1e0*user_xr_ion(E), H --> Hk + E  #5
    3.7e-2*user_crate, H2 --> H + Hk + E  #6
    1.e0*user_uv_ion, H --> Hk + E   #7
    ###### reactions listed for H2 formation/destruction MINUS!!!! those already listed before
    4.4886e-9*Tgas^0.109127e0*exp(-1.01858e5*1/Tgas), H2 + E --> H + H + E  #8
    # the following formula for h2_high_n is not entirely correct as it is only valid for T<300K
    T_interval(Tgas, 0, 3e2)*1.1e-9*Tgas^0.135e0*exp(-5.2e4*1/Tgas)*ch3(Tgas)^h2var0(H, H2, Hk, Tgas), H2 + H --> H + H + H  #9
    T_interval(Tgas, 3e2+0.1, 1e10)*3.7e-8*1/Tgas^0.485e0*exp(-5.2e4*1/Tgas)*ch3(Tgas)^h2var0(H, H2, Hk, Tgas), H2 + H --> H + H + H  #10
    6.5e-8*exp(-5.2e4*1/Tgas)*1/Tgas^0.485e0*ch4(Tgas)^h2var0(H, H2, Hk, Tgas), H2 + H2 --> H + H + H + H  #11
    #H2 ON GRAINS WITH RATE APPROXIMATION
    user_dust_to_gas_ratio*user_deff*3e-18*sqrt(Tgas)*fA/(1e0+0.04e0*sqrt(Tgas+user_Tdust)+0.002e0*Tgas+8e-6*Tgas^2.0)*H_nuclei(H, H2, Hk)/max(H,1e-40), H + H  --> H2  #12
    3.3e-11*user_Ghab*exp(-3.5e0*user_Av)*user_H2self, H2 --> H + H  #13
    # fudge factor to get HI production right, as it is actually H2->H2+ + e-
    0.5e0*2e0*user_crate, H2 --> H + H  #14
    2.2e-1*user_crate, H2 --> H + H  #15
    1e0*user_uv_ion, H2 --> H + H  #16
    ###### reactions listed for CO formation/destruction
    5e-16*beta(O)*1e0/(max(E,1e-40)*max(O, 1e-40)), Ck + E + H2 + O --> CO + H + H  #17
    1.235e-10*user_Ghab*exp(-2.5e0*user_Av)*user_COself, CO --> Ck + O + E  #18
    3.861003861e0*user_uv_ion, CO --> Ck + O + E  #19

end Tgas user_crate user_uv_ion user_dust_to_gas_ratio user_deff fA user_Tdust user_Ghab user_Av user_H2self user_COself
