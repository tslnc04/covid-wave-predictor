const svg = d3.select('#map')

// create an element to show up when hovering over counties
const hover = d3
    .select('body')
    .append('div')
    .attr('id', 'hover')
    .style('opacity', 0.9)
    .style('display', 'none')

// use D3 to handle the map and projection
const proj = d3.geoAlbersUsa().scale(1300).translate([487.5, 305])
const path = d3.geoPath(proj)

// create a color ramp from Yellow -> Orange -> Red
const color = d3
    .scaleThreshold()
    .domain(d3.range(0, 1, 0.125))
    .range(d3.schemeYlOrRd[9])

// add a group to the svg that allows for zooms
const g = svg.append('g')

// handle transforms
svg.call(
    d3.zoom().on('zoom', ({ transform }) => g.attr('transform', transform))
)

Promise.all([d3.json('county_predictions.json'), d3.json('county_map.json')])
    .then((data) => {
        const predictions = data[0]
        const us = data[1]

        g.selectAll('path')
            .data(
                topojson.feature(us, us.objects.cb_2018_us_county_20m).features
            )
            .enter()
            .append('path')
            // set the color of the county based on the prediction and using the
            // color ramp
            .attr('fill', (d) => {
                const data_point = predictions.filter(
                    (county) => county.fips === d.properties.GEOID
                )
                return data_point[0] ? color(data_point[0].prediction) : 0
            })
            .attr('d', path)
            // show the box with information on hover
            .on('mouseover', (e, d) => {
                hover.style('display', 'block')
                hover
                    .html(() => {
                        const data_point = predictions.filter(
                            (county) => county.fips === d.properties.GEOID
                        )
                        return data_point[0]
                            ? `<span style="color: ${color(
                                  data_point[0].prediction
                              )};">&#x2588;&#x2588;&nbsp;</span>${
                                  data_point[0].county
                              }, ${data_point[0].state}<br/>${(
                                  data_point[0].prediction * 100
                              ).toFixed(2)}% chance of wave`
                            : ''
                    })
                    .style('left', `${e.pageX + 10}px`)
                    .style('top', `${e.pageY - 28}px`)
            })
            // get rid of the box when it's no longer hovered over
            .on('mouseout', () => hover.style('display', 'none'))

        // add the counties to the datalist element so that the input has
        // autocomplete for each county
        d3.select('#counties')
            .selectAll('option')
            .data(predictions)
            .enter()
            // remove values with unknown fips, usually cities with their data
            // separate in the source data
            .filter((d) => d.fips !== '')
            .append('option')
            .attr('value', (d) => d.fips)
            .text((d) =>
                d.county === 'Unknown'
                    ? `${d.state}`
                    : `${d.county}, ${d.state}`
            )

        // display the same data as when hovering upon lookup
        d3.select('#submit-lookup').on('click', () => {
            const fips = d3.select('#lookup').node().value
            const data_point = predictions.filter(
                (county) => fips && county.fips === fips
            )

            if (data_point.length === 0) {
                return
            }

            d3.select('#lookup-output').html(
                () =>
                    `<span style="color: ${color(
                        data_point[0].prediction
                    )};">&#x2588;&#x2588;&nbsp;</span>${
                        data_point[0].county
                    }, ${data_point[0].state}<br/>${(
                        data_point[0].prediction * 100
                    ).toFixed(2)}% chance of wave`
            )
        })
    })
    .catch((err) => console.log(err))
