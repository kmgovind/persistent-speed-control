module SolarInsolationModel

export SolarInsolation

"""
Get angle representing day of year

Inputs:
    dayOfYear [days]
Returns:
    dayOfYear as angle [rad]

Reference:
    Hulstrom, Solar Resources, Eq. 3.2
"""
function DayOfYearToAngle(dayOfYear)
    return 2π * dayOfYear/365
end

"""
Get the distance to the sun, normalized

Inputs:
    dayOfYear [days]

Returns:
    R/R0: normalized distance to sun

Note:
    R0: 1.496e8 km

Reference:
    Hulstrom, Solar Resources, Eq. 3.3
"""
function DistanceToSun(dayOfYear)
    TH = DayOfYearToAngle(dayOfYear)
    RR0_sq = 1.000110 + 0.034221*cos(TH) + 0.001280sin(TH) + 0.000719cos(2TH) + 0.000077sin(2TH)
    return sqrt(RR0_sq)
end


"""
Equation of time

Inputs:
    dayOfYear [days]

Returns:
    correction between standard time and local solar time [mins]

Reference:
    Hulstrom, Solar Resources, Eq. 3.4
"""
function EqOfTime(dayOfYear)
    TH = DayOfYearToAngle(dayOfYear)
    return 0.0172 + 0.4281cos(TH) - 7.3515sin(TH) - 3.3495cos(2TH) - 9.3619cos(2TH)
end

"""
Declination of the Sun [rad]

Inputs:
    dayOfYear [days]

Returns:
    Declination of the sun [rad]

Reference:
    Hulstrom, Solar Resources, Eq. 3.5
"""
function Declination(dayOfYear)
    TH = DayOfYearToAngle(dayOfYear)
    return  (π/180) * (0.39637 - 22.91326cos(TH) + 4.02543sin(TH) - 0.38720cos(2TH) + 0.05197sin(2TH) - 0.15453cos(3TH) + 0.08479sin(3TH))
end


"""
convert from standard time to solar time

Inputs:
    dayOfYear [days]
    localStandardTime [hours]
    lon = longitude [deg]
    refLon = reference longitude [deg]


Returns:
    true solar time [hours]

Reference:
    Hulstrom, Solar Resources, Eq. 3.6
"""
function TrueSolarTime(dayOfYear, localStandardTime, lon, refLon)
    TH = DayOfYearToAngle(dayOfYear)
    ET = EqOfTime(TH)
    return localStandardTime + (lon - refLon) / 15 + ET/60
end


"""
returns the hour angle based on the true solar time

Inputs:
    trueSolarTime [hours]

Returns:
    hourAngle [rad]

Reference:
    Hulstrom, Solar Resources, Eq. 3.7
"""
function HourAngle(trueSolarTime)
    return (π/180) *  15 * (trueSolarTime - 12)
end

"""
returns the sun's elevation angle

Inputs:
    dayOfYear: [days]
    solarTime: [hours]
    latitude: [deg]

Returns:
    elevation angle [rad]

Reference:
    Hulstrom, Solar Resources, Eq. 3.8
"""
function Elevation(dayOfYear, solarTime, lat)
    dec_rad = Declination(dayOfYear)
    lat_rad = π/180 * lat
    hourAngle_rad = HourAngle(solarTime)
    return asin( sin(lat_rad) * sin(dec_rad) + cos(lat_rad) * cos(dec_rad) * cos(hourAngle_rad))
end



"""
returns the sun's azimuth angle

Inputs:
    dayOfYear: [days]
    solarTime: [hours]
    latitude: [deg]

Returns:
    azimuth angle [rad]

Reference:
    Hulstrom, Solar Resources, Eq. 3.10
"""
function Azimuth(dayOfYear, solarTime, lat)
    
    TH = DayOfYearToAngle(dayOfYear) 
    dec_rad = Declination(dayOfYear)
    hourAngle_rad = HourAngle(solarTime)
    EL_rad = Elevation(dayOfYear, hourAngle_rad, lat)
    return asin(cos(dec_rad)*sin(hourAngle_rad) / cos(EL_rad))
end


"""
returns the sun's zenith angle

Inputs:
    dayOfYear: [days]
    solarTime: [hours]
    latitude: [deg]

Returns:
    zenith angle [rad]

Reference:
    Hulstrom, Solar Resources, Eq. 3.9
"""
function ZenithAngle(dayOfYear, solarTime, lat)
    return π/2 - Elevation(dayOfYear, solarTime, lat)
end


"""
returns the solar insolation [kW/m^2]

Inputs:
    dayOfYear: [days]
    solarTime: [hours]
    latitude: [deg]

Returns:
    solarInsolation: [kW/m^2]

Reference:
    Hulstrom, Solar Resources, Eq. 3.11-12
"""
function SolarInsolation(dayOfYear, solarTime, lat; D0=1.377)
    
    RR0_sq = DistanceToSun(dayOfYear)^2
    D = D0 / RR0_sq
    z_rad = ZenithAngle(dayOfYear, solarTime, lat)
    return D * cos(z_rad)
end
    

end

# dayOfYear = 0.0
# lat = 35.0 # degrees
# plot(solarTime -> SolarInsolationModel.SolarInsolation(dayOfYear, solarTime, lat), 0, 24, linestyle=:dash)
# plot!(solarTime -> max(0, SolarInsolationModel.SolarInsolation(dayOfYear, solarTime, lat)), 0, 24, label="Day $(dayOfYear), Lat: $(lat) deg")
# xlabel!("Solar Time [hours, 12=noon]")
# ylabel!("Solar Insolation [kW / m2]")
# plot()
# for dayOfYear in [0, 90, 180, 270], lat in [0, 35, 60, 90]
#     plot!(solarTime -> max(0, SolarInsolationModel.SolarInsolation(dayOfYear, solarTime, lat)), 0, 24, label="Day $(dayOfYear), Lat: $(lat)o")
# end
# xlabel!("Solar Time [hours, 12=noon]")
# ylabel!("Solar Insolation [kW / m2]")
# plot()
# for dayOfYear in [0, 90, 180, 270]
#     lat = 35.0
#     plot!(solarTime -> max(0, SolarInsolationModel.SolarInsolation(dayOfYear, solarTime, lat)), 0, 24, label="Day $(dayOfYear), Lat: $(lat)o")
# end
# xlabel!("Solar Time [hours, 12=noon]")
# ylabel!("Solar Insolation [kW / m2]")
# @gif for dayOfYear = 0:5:365
#     lat = 35.0 # degrees
#     plot(solarTime -> SolarInsolationModel.SolarInsolation(dayOfYear, solarTime, lat), 0, 24, linestyle=:dash, label=false)
#     plot!(solarTime -> max(0, SolarInsolationModel.SolarInsolation(dayOfYear, solarTime, lat)), 0, 24, label="Day $(dayOfYear), Lat: $(lat) deg")
#     xlabel!("Solar Time [hours, 12=noon]")
#     ylabel!("Solar Insolation [kW / m2]")
#     ylims!(-1.4, 1.4)
# end
# plot()
# for dayOfYear in [0, 90, 180, 270]
#     lat = 35.0
#     plot!(solarTime -> max(0, SolarInsolationModel.SolarInsolation(dayOfYear, solarTime, lat)), 0, 3*24, label="Day $(dayOfYear), Lat: $(lat)o")
# end
# xlabel!("Solar Time [hours, 12=noon]")
# ylabel!("Solar Insolation [kW / m2]")
# dayGif = @animate for dayOfYear = 0:5:365
# # dayOfYear = 0
    
#     plots = [ begin
#     plot(solarTime -> SolarInsolationModel.SolarInsolation(dayOfYear, solarTime, lat), 0, 24, linestyle=:dash, label=false)
#     plot!(solarTime -> max(0, SolarInsolationModel.SolarInsolation(dayOfYear, solarTime, lat)), 0, 24, label="Day $(dayOfYear), Lat: $(lat) deg")
#     xlabel!("Solar Time [h, 12=noon]")
#     ylabel!("Insolation [kW / m2]")
#     ylims!(-1.4, 1.4)
#         title!("Day: $(dayOfYear), Lat: $(lat)deg")
#         plot!(legend=false)
#     end for lat in [0, 30, 60, 80] ]

# plot(plots...)

#         if dayOfYear == 180
#             savefig("fig/solarinsolation_day180.png")
#         end
# end
# gif(dayGif, "solar_insolation.gif")
# ## SIMPLIFIED MODEL

# el = asin( sin(lat_rad) * sin(dec_rad) + cos(lat_rad) * cos(dec_rad) * cos(hourAngle_rad))
# z = π/2 - el
# D0 cos(z)

# # therefore cos(z) = 
# Cos[dec] Cos[hourAngle] Cos[lat] + Sin[dec] Sin[lat]
# function solar_insolation(hourAngle, lat, dec)
#     return sin(dec)*sin(lat) + cos(dec) * cos(lat) * cos(hourAngle)
# end
