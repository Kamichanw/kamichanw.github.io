# frozen_string_literal: true

require 'nokogiri'
require 'pathname'
require 'set'

Jekyll::Hooks.register :polyglot, :post_write do |site|
  destination = Pathname(site.dest)
  sitemap_path = destination.join('sitemap.xml')
  next unless sitemap_path.file?

  sitemap = Nokogiri::XML(sitemap_path.read)
  urlset = sitemap.at_xpath('/xmlns:urlset')
  next unless urlset

  known_urls = sitemap.xpath('//xmlns:loc').map(&:text).to_set
  site_url = "#{site.config.fetch('url')}#{site.config.fetch('baseurl', '')}".delete_suffix('/')

  site.languages.each do |language|
    next if language == site.default_lang

    language_root = destination.join(language)
    language_root.glob('**/*.html').each do |html_path|
      html = Nokogiri::HTML(html_path.read)
      next if html.at_css('meta[http-equiv="refresh"]')

      canonical = html.at_css('link[rel="canonical"]')&.[]('href')
      next unless canonical&.start_with?("#{site_url}/#{language}/")
      next unless known_urls.add?(canonical)

      url = Nokogiri::XML::Node.new('url', sitemap)
      url.namespace = urlset.namespace
      loc = Nokogiri::XML::Node.new('loc', sitemap)
      loc.namespace = urlset.namespace
      loc.content = canonical
      url.add_child(loc)
      urlset.add_child(url)
    end

    language_sitemap = language_root.join('sitemap.xml')
    language_sitemap.delete if language_sitemap.file?
  end

  sitemap_path.write(sitemap.to_xml)
end
